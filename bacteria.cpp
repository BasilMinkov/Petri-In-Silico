#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <uuid/uuid.h>
#include <GL/glut.h>
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <thread>
#include <omp.h> // For parallel processing

using namespace std;
using namespace Eigen;

// Constants
const int MAXIMUM = 95;
const int MINIMUM = 5;
const int WIDTH = 1000;
const int HEIGHT = 1000;
const int GRID_SIZE = 100;
const float DOT_RADIUS = 0.02;
const int INITIAL_ENERGY = 100;
const int N_BACTERIA = 1000;
const int NUM_LAYERS = 7;
const int NEURONS_PER_LAYER = 7;
const int INPUT_SIZE = 5;
const int OUTPUT_CLASSES = 6;
const int MOVE_ENERGY = -1;
const int EAT_ENERGY = -5;
const int DIVIDE_ENERGY = -10;
const int RIGHT_ENERGY = 0;
const int LEFT_ENERGY = 0;
const int SYNTHESIS_ENERGY = 2;

vector<string> ACTIONS = {"right", "left", "eat", "move", "divide", "synthesise"};

vector<vector<int>> SIGHTS = {
    {1, 0},
//    {1, 1},
    {0, 1},
//    {-1, 1},
    {-1, 0},
//    {-1,-1},
    {0, -1},
//    {1, -1}
};

vector<string> LAYERS = {"l1", "l2", "l3", "l4", "l5", "l6"};
vector<string> ACTIVATIONS = {"n1", "n2", "n3", "n4", "n5", "n6"};

// Activation functions
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double random_neuron(double x) {
    return static_cast<double>(rand()) / RAND_MAX;
}

typedef double (*ActivationFunc)(double);

vector<ActivationFunc> activation_functions = {sigmoid, random_neuron};

double apply_activation(int neuron_type, double x) {
    ActivationFunc func = activation_functions[neuron_type];
    return func(x);
}

// Helper functions
void draw_dot(float y, float x, vector<float> color, int size = 10) {
    glPointSize(size);
    glColor3f(color[0], color[1], color[2]);
    glBegin(GL_POINTS);
    glVertex2f(x, y);
    glEnd();
}

vector<int> from_grid_coordinates(int i, int j) {
    return {static_cast<int>((i + 0.5) / GRID_SIZE), static_cast<int>((j + 0.5) / GRID_SIZE)};
}

vector<int> to_grid_coordinates(int i, int j) {
    return {static_cast<int>((GRID_SIZE * i) - 0.5), static_cast<int>((GRID_SIZE * i) - 0.5)};
}

const double p = 1.0 / 20;
const int n_features = INPUT_SIZE * NEURONS_PER_LAYER + NEURONS_PER_LAYER + NEURONS_PER_LAYER * NEURONS_PER_LAYER * 5 + NEURONS_PER_LAYER * 4 + OUTPUT_CLASSES * NEURONS_PER_LAYER + OUTPUT_CLASSES;

class Bacteria {
public:
    uuid_t idb;
    int time;
    double energy;
    double energy_from_predation;
    double energy_from_synthesis;
    double r, g, b;
    bool dead;
    bool acted;
    int sight;

    MatrixXd l1, l2, l3, l4, l5, l6;
    VectorXi n1, n2, n3, n4, n5, n6;

    Bacteria(int time = 0, double energy = INITIAL_ENERGY)
        : time(time), energy(energy), energy_from_predation(0), energy_from_synthesis(0), dead(false), acted(false) {
        uuid_generate(idb);
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(0.0, 1.0);
        r = dis(gen);
        g = dis(gen);
        b = dis(gen);
        sight = rand() % SIGHTS.size();

        l1 = MatrixXd::Random(INPUT_SIZE, NEURONS_PER_LAYER);
        n1 = VectorXi::Zero(NEURONS_PER_LAYER).unaryExpr([](int) { return rand() % 2; });

        l2 = MatrixXd::Random(NEURONS_PER_LAYER, NEURONS_PER_LAYER);
        n2 = VectorXi::Zero(NEURONS_PER_LAYER).unaryExpr([](int) { return rand() % 2; });

        l3 = MatrixXd::Random(NEURONS_PER_LAYER, NEURONS_PER_LAYER);
        n3 = VectorXi::Zero(NEURONS_PER_LAYER).unaryExpr([](int) { return rand() % 2; });

        l4 = MatrixXd::Random(NEURONS_PER_LAYER, NEURONS_PER_LAYER);
        n4 = VectorXi::Zero(NEURONS_PER_LAYER).unaryExpr([](int) { return rand() % 2; });

        l5 = MatrixXd::Random(NEURONS_PER_LAYER, NEURONS_PER_LAYER);
        n5 = VectorXi::Zero(NEURONS_PER_LAYER).unaryExpr([](int) { return rand() % 2; });

        l6 = MatrixXd::Random(NEURONS_PER_LAYER, OUTPUT_CLASSES);
        n6 = VectorXi::Zero(OUTPUT_CLASSES).unaryExpr([](int) { return rand() % 2; });
    }

    string act(VectorXd x) {
        for (int i = 0; i < LAYERS.size(); ++i) {
            string layer = LAYERS[i];
            string activation = ACTIVATIONS[i];
            MatrixXd* layer_matrix = nullptr;
            VectorXi* activation_vector = nullptr;

            if (layer == "l1") layer_matrix = &l1;
            else if (layer == "l2") layer_matrix = &l2;
            else if (layer == "l3") layer_matrix = &l3;
            else if (layer == "l4") layer_matrix = &l4;
            else if (layer == "l5") layer_matrix = &l5;
            else if (layer == "l6") layer_matrix = &l6;

            if (activation == "n1") activation_vector = &n1;
            else if (activation == "n2") activation_vector = &n2;
            else if (activation == "n3") activation_vector = &n3;
            else if (activation == "n4") activation_vector = &n4;
            else if (activation == "n5") activation_vector = &n5;
            else if (activation == "n6") activation_vector = &n6;

            x = x.transpose() * (*layer_matrix);
            for (int j = 0; j < x.size(); ++j) {
                x(j) = apply_activation((*activation_vector)(j), x(j));
            }
        }
        int y = distance(x.data(), max_element(x.data(), x.data() + x.size()));
        return ACTIONS[y];
    }

    Bacteria* divide() {
        Bacteria* copied_obj = new Bacteria(*this);
        energy = floor(energy / 2);
        uuid_generate(copied_obj->idb);
        copied_obj->energy = floor(copied_obj->energy / 2);
        copied_obj->energy_from_predation = 0;
        copied_obj->energy_from_synthesis = 0;
        copied_obj->time = 0;
        return copied_obj;
    }

    void change_color() {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> d(0, 0.05);
        r = min(1.0, max(0.0, r + d(gen)));
        g = min(1.0, max(0.0, g + d(gen)));
        b = min(1.0, max(0.0, b + d(gen)));
    }

    void mutate() {

        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(0, n_features - 1);
        vector<string> all_attributes = LAYERS;
        all_attributes.insert(all_attributes.end(), ACTIVATIONS.begin(), ACTIVATIONS.end());

        float thre = static_cast<double>(rand()) / RAND_MAX;

        cout << thre << endl;
        if (thre > 0.5){

        #pragma omp parallel for
        for (const string& name : all_attributes) {
            if (name[0] == 'l') {
                MatrixXd* value = nullptr;
                if (name == "l1") value = &l1;
                else if (name == "l2") value = &l2;
                else if (name == "l3") value = &l3;
                else if (name == "l4") value = &l4;
                else if (name == "l5") value = &l5;
                else if (name == "l6") value = &l6;
                for (int i = 0; i < value->size(); ++i) {
                    if (dis(gen) == 0) {
                        change_color();
                        value->coeffRef(i) = static_cast<double>(rand()) / RAND_MAX;
                    }
                }
            } else if (name[0] == 'n') {
                VectorXi* value = nullptr;
                if (name == "n1") value = &n1;
                else if (name == "n2") value = &n2;
                else if (name == "n3") value = &n3;
                else if (name == "n4") value = &n4;
                else if (name == "n5") value = &n5;
                else if (name == "n6") value = &n6;

                #pragma omp parallel for
                for (int i = 0; i < value->size(); ++i) {
                    if (dis(gen) == 0) {
                        change_color();
                        value->coeffRef(i) = rand() % 2;
                    }
                }
            }
        }
        }
    }

    bool are_relatives(Bacteria* other) {
        double distance = std::sqrt(std::pow(r - other->r, 2) + std::pow(g - other->g, 2) + std::pow(b - other->b, 2));
        return distance < 0.2;
    }

//    bool are_relatives(Bacteria* other) {
//        // Use bitwise operations to quickly compare the activation vectors
//        #pragma omp parallel for
//        for (int i = 0; i < LAYERS.size(); ++i) {
//            MatrixXd* layer1 = nullptr;
//            MatrixXd* layer2 = nullptr;
//            if (LAYERS[i] == "l1") layer1 = &l1, layer2 = &other->l1;
//            else if (LAYERS[i] == "l2") layer1 = &l2, layer2 = &other->l2;
//            else if (LAYERS[i] == "l3") layer1 = &l3, layer2 = &other->l3;
//            else if (LAYERS[i] == "l4") layer1 = &l4, layer2 = &other->l4;
//            else if (LAYERS[i] == "l5") layer1 = &l5, layer2 = &other->l5;
//            else if (LAYERS[i] == "l6") layer1 = &l6, layer2 = &other->l6;
//            if ((layer1->array() == layer2->array()).count() > 20) {
//                return true;
//            }
//        }
//        #pragma omp parallel for
//        for (int i = 0; i < ACTIVATIONS.size(); ++i) {
//            VectorXi* act1 = nullptr;
//            VectorXi* act2 = nullptr;
//            if (ACTIVATIONS[i] == "n1") act1 = &n1, act2 = &other->n1;
//            else if (ACTIVATIONS[i] == "n2") act1 = &n2, act2 = &other->n2;
//            else if (ACTIVATIONS[i] == "n3") act1 = &n3, act2 = &other->n3;
//            else if (ACTIVATIONS[i] == "n4") act1 = &n4, act2 = &other->n4;
//            else if (ACTIVATIONS[i] == "n5") act1 = &n5, act2 = &other->n5;
//            else if (ACTIVATIONS[i] == "n6") act1 = &n6, act2 = &other->n6;
//            if ((act1->array() == act2->array()).count() > 20) {
//                return true;
//            }
//        }
//
//        return false;
//    }
};

void mainLoop() {
    // Placeholder function for GLUT's main loop
}

void mainFunction(vector<Bacteria>& genomes) {
    int argc = 1;
    char *argv[1] = {(char*)"Something"};
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Bacteria Simulation");

    glOrtho(0, WIDTH, HEIGHT, 0, -1, 1);

    vector<vector<Bacteria*>> field(GRID_SIZE, vector<Bacteria*>(GRID_SIZE, nullptr));
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, GRID_SIZE * GRID_SIZE - 1);
    vector<int> random_positions(N_BACTERIA);
    generate(random_positions.begin(), random_positions.end(), [&]() { return dis(gen); });

    for (int pos : random_positions) {
        int x = pos % GRID_SIZE;
        int y = pos / GRID_SIZE;
        field[y][x] = new Bacteria();
    }

    for (int i = 0; i < MINIMUM; ++i) {
        fill(field[i].begin(), field[i].end(), nullptr);
        for (int j = 0; j < GRID_SIZE; ++j) {
            field[j][i] = nullptr;
            field[j][GRID_SIZE - 1 - i] = nullptr;
        }
        fill(field[GRID_SIZE - 1 - i].begin(), field[GRID_SIZE - 1 - i].end(), nullptr);
    }

    bool running = true;
    bool break_process = false;  // Changed to false to prevent premature exit
    double daytime = 1;
    bool up = false;
    string mode = "gene";
    uuid_t chosen_bacteria;
    bool chosen_bacteria_exists = false;

    int timer = 0;
    bool show = true;

    while (running) {
        timer++;

        vector<string> memory;
        int nb = 0;
        double common_energy = 0;

//        if (up) {
//            daytime += 0.01;
//        } else {
//            daytime -= 0.01;
//        }

        if (daytime >= 1) {
            up = false;
        }
        if (daytime <= 0) {
            up = true;
        }

        if (show) {
            glClearColor(daytime / 10, daytime / 10, daytime / 10, 1);
            glClear(GL_COLOR_BUFFER_BIT);
        }

        // Handle events (e.g., keyboard/mouse inputs)
        // Placeholder: Implement event handling

        #pragma omp parallel for collapse(2)
        for (int iy = 0; iy < GRID_SIZE; ++iy) {
            for (int ix = 0; ix < GRID_SIZE; ++ix) {
                if (field[iy][ix] != nullptr) {
                    field[iy][ix]->acted = false;
                }
            }
        }

        #pragma omp parallel for collapse(2) reduction(+:common_energy, nb)
        for (int iy = 0; iy < GRID_SIZE; ++iy) {
            for (int ix = 0; ix < GRID_SIZE; ++ix) {
                if (field[iy][ix] == nullptr || field[iy][ix]->acted || field[iy][ix]->dead) {
                    continue;
                }

                common_energy += field[iy][ix]->energy;
                nb++;

                field[iy][ix]->time += 1;

                if (field[iy][ix]->energy <= 1 || field[iy][ix]->time >= 1000) {
                    delete field[iy][ix];
                    field[iy][ix] = nullptr;
                    continue;
                }

                VectorXd features(5);
                int ny = iy + SIGHTS[field[iy][ix]->sight][0];
                int nx = ix + SIGHTS[field[iy][ix]->sight][1];

                if (ny < 0 || nx < 0 || ny >= GRID_SIZE || nx >= GRID_SIZE || field[ny][nx] == nullptr) {
                    features[0] = 1;
                    features[1] = 0;
                } else if (field[ny][nx] != nullptr) {
                    features[0] = 0;
//                    features[1] = 1;
                    if (field[iy][ix]->are_relatives(field[ny][nx])) {
                        features[1] = 1;
                    } else {
                        features[1] = 0;
                    }
                }

                features[2] = SIGHTS[field[iy][ix]->sight][0];
                features[3] = SIGHTS[field[iy][ix]->sight][1];
                features[4] = field[iy][ix]->energy / 1000.0;
                string action = field[iy][ix]->act(features);

                field[iy][ix]->acted = true;
                bool moved = false;

                if (action == "move") {
                    field[iy][ix]->energy += MOVE_ENERGY;
                    int ny = iy + SIGHTS[field[iy][ix]->sight][0];
                    int nx = ix + SIGHTS[field[iy][ix]->sight][1];
                    if (ny >= 0 && nx >= 0 && ny < GRID_SIZE && nx < GRID_SIZE && field[ny][nx] == nullptr) {
                        field[ny][nx] = field[iy][ix];
                        field[iy][ix] = nullptr;
                        moved = true;
                    }
                }

                if (action == "right") {
                    field[iy][ix]->energy += RIGHT_ENERGY;
                    field[iy][ix]->sight = (field[iy][ix]->sight + 1) % SIGHTS.size();
                }

                if (action == "left") {
                    field[iy][ix]->energy += LEFT_ENERGY;
                    field[iy][ix]->sight = (field[iy][ix]->sight - 1 + SIGHTS.size()) % SIGHTS.size();
                }

                if (action == "eat") {
                    field[iy][ix]->energy += EAT_ENERGY;
                    int ny = iy + SIGHTS[field[iy][ix]->sight][0];
                    int nx = ix + SIGHTS[field[iy][ix]->sight][1];
                    if (ny >= 0 && nx >= 0 && ny < GRID_SIZE && nx < GRID_SIZE && field[ny][nx] != nullptr) {
                        if (typeid(*field[ny][nx]) == typeid(Bacteria)) {
                            field[iy][ix]->energy += field[ny][nx]->energy;
                            delete field[ny][nx];
                            field[ny][nx] = field[iy][ix];
                            field[iy][ix] = nullptr;
                            moved = true;
                        }
                    }
                }

                if (action == "divide") {
                    field[iy][ix]->energy += DIVIDE_ENERGY;
                    int ny = iy + SIGHTS[field[iy][ix]->sight][0];
                    int nx = ix + SIGHTS[field[iy][ix]->sight][1];
                    if (ny >= 0 && nx >= 0 && ny < GRID_SIZE && nx < GRID_SIZE && field[ny][nx] == nullptr) {
                        field[ny][nx] = field[iy][ix]->divide();
                        field[ny][nx]->mutate();
                        moved = true;
                        break;
                    }
                }

                if (action == "synthesise") {
                    if (daytime > 0.5) {
                        field[iy][ix]->energy += SYNTHESIS_ENERGY;
                        field[iy][ix]->energy_from_synthesis += SYNTHESIS_ENERGY;
                        moved = true;
                    }
                }

//                if (nb < 10 && break_process) {
//                    running = false;
//                }
            }
        }

        if (show) {
            #pragma omp parallel for collapse(2)
            for (int iy = 0; iy < GRID_SIZE; ++iy) {
                for (int ix = 0; ix < GRID_SIZE; ++ix) {
                    if (field[iy][ix] != nullptr) {
                        float by = iy * 10;
                        float bx = ix * 10;
                        if (mode == "gene") {
                            draw_dot(by, bx, {static_cast<float>(field[iy][ix]->r), static_cast<float>(field[iy][ix]->g), static_cast<float>(field[iy][ix]->b)});
                        }
                    }
                }
            }
        }

        if (show) {
            glutSwapBuffers();
        }

        // Debug output
        cout << "Timer: " << timer << ", Number of Bacteria: " << nb << ", Common Energy: " << common_energy << endl;

//        if (nb < 100) {
//            cout << "Less than 100 bacteria left. Exiting." << endl;
//            running = false;
//        }
    }

    for (int iy = 0; iy < GRID_SIZE; ++iy) {
        for (int ix = 0; ix < GRID_SIZE; ++ix) {
            delete field[iy][ix];
        }
    }
}

int main(int argc, char** argv) {
    vector<Bacteria> genomes;
    mainFunction(genomes);
    return 0;
}
