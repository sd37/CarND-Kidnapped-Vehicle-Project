/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    this->num_particles = 5;

    for (int i = 0; i < this->num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        this->particles.push_back(p);
        this->weights.push_back(p.weight);
    }

    this->is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    double current_velocity_meas = velocity;
    double current_yaw_rate_meas = yaw_rate;

    double new_x = -1.0;
    double new_y = -1.0;
    double new_theta = -1.0;

    default_random_engine gen;

    for (int i = 0; i < this->num_particles; i++) {
        // use motion model equations for prediction.
        if (yaw_rate < 0.001) {
            new_x = particles[i].x + current_velocity_meas * delta_t * cos(particles[i].theta);
            new_y = particles[i].y + current_velocity_meas * delta_t * sin(particles[i].theta);
            new_theta = particles[i].theta;
        } else {
            new_x = particles[i].x + (current_velocity_meas / current_yaw_rate_meas) *
                                     (sin(particles[i].theta + current_yaw_rate_meas * delta_t) -
                                      sin(particles[i].theta));

            new_y = particles[i].y + (current_velocity_meas / current_yaw_rate_meas) *
                                     (cos(particles[i].theta) -
                                      cos(particles[i].theta + current_yaw_rate_meas * delta_t));

            new_theta = particles[i].theta + current_yaw_rate_meas * delta_t;

        }

        // update particles estimate. Add gaussian noise as well.

        particles[i].x = normal_distribution<double>(new_x, std_pos[0])(gen);
        particles[i].y = normal_distribution<double>(new_y, std_pos[1])(gen);
        particles[i].theta = normal_distribution<double>(new_theta, std_pos[2])(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    for (auto &obsv : observations) {
        double closest_lm_d = -1.0;
        int closest_lm_id = -1;
        for (auto &lm : predicted) {
            double cal_d = dist(lm.x, lm.y, obsv.x, obsv.y);
            if (closest_lm_d > cal_d) {
                closest_lm_d = cal_d;
                closest_lm_id = lm.id;
            }
        }

        obsv.id = closest_lm_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
    //   for the fact that the map's y-axis actually points downwards.)
    //   http://planning.cs.uiuc.edu/node99.html

    vector<LandmarkObs> transformed_observations;
    vector<LandmarkObs> predicted;

    for (auto &p: particles) {
        // Step1 : Transform observations to Global Map Coordinate system wrt to a predicted vehicle particle.
        // https://discussions.udacity.com/t/coordinate-transform/241288/4

        transformed_observations.clear();
        predicted.clear();

        for (int i = 0; i < observations.size(); i++) {
            LandmarkObs tmp;
            tmp.x = p.x + observations[i].x * cos(p.theta) - observations[i].y * sin(p.theta);
            tmp.y = p.y + observations[i].x * sin(p.theta) + observations[i].y * cos(p.theta);
            tmp.id = -1;
            transformed_observations.push_back(tmp);
        }

        // Step 2: Construct predicted vector. Remove landmarks which are not in sensor range.
        // https://discussions.udacity.com/t/why-does-updateweights-function-needs-sensor-range/248695

        double xmin_range = p.x - sensor_range;
        double xmax_range = p.x + sensor_range;
        double ymin_range = p.y - sensor_range;
        double ymax_range = p.y + sensor_range;

        for (auto &lm: map_landmarks.landmark_list) {
            if (xmin_range < lm.x_f && lm.x_f < xmax_range &&
                ymin_range < lm.y_f && lm.y_f < ymax_range) {
                LandmarkObs tmp_lm;
                tmp_lm.x = lm.x_f;
                tmp_lm.y = lm.y_f;
                tmp_lm.id = lm.id_i;
                predicted.push_back(tmp_lm);
            }
        }

        // Step 3: Do dataAssociation and map each TOBS to a landmark id.

        dataAssociation(predicted, transformed_observations);

        // Step 4: update the weights of all the particles using multivariate gaussian distribution.

        double new_particle_weight = 1.0;

        for (auto &obsv: transformed_observations) {
            double mvg_prob = 0;

            double tmp_x = obsv.x;
            double tmp_y = obsv.y;

            Map::single_landmark_s cl_lm = map_landmarks.landmark_list[obsv.id - 1];

            double mu_x = cl_lm.x_f;
            double mu_y = cl_lm.y_f;

            double sigma_x = std_landmark[0];
            double sigma_y = std_landmark[1];

            double p1 = 1 / (2.0 * M_PI * sigma_x * sigma_y);
            double p2x = ((tmp_x - mu_x) * (tmp_x - mu_x)) / (2.0 * sigma_x * sigma_x);
            double p2y = ((tmp_y - mu_y) * (tmp_y - tmp_y)) / (2.0 * sigma_y * sigma_y);
            double p2 = exp(-1 * (p2x + p2y));
            mvg_prob = p1 * p2;

            new_particle_weight *= mvg_prob;
        }

        p.weight = new_particle_weight;

    }

}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

void ParticleFilter::write(std::string filename) {
    // You don't need to modify this file.
    std::ofstream dataFile;
    dataFile.open(filename, std::ios::app);
    for (int i = 0; i < num_particles; ++i) {
        dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
    }
    dataFile.close();
}
