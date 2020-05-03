/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine gen;
  
  num_particles = 100;  // Set the number of particles
  normal_distribution<double> dist_x(x, std[0]); // Set gaussian distribution for x with std_x
  normal_distribution<double> dist_y(y, std[1]); // Set gaussian distribution for y with std_y
  normal_distribution<double> dist_theta(theta, std[2]); // Set gaussian distribution for theta with std_theta
  
  // Create num_particles
  for (int id = 0; id < num_particles; ++id) {
    /*# struct Particle {
    #   int id;
    #   double x;
    #   double y;
    #   double theta;
    #   double weight;
    #   std::vector<int> associations;
    #   std::vector<double> sense_x;
    #   std::vector<double> sense_y;
    # }*/
    
    // Create particle with weight 1 using samples from our gaussian distributions.
    Particle particle = {id, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0};
    particles.push_back(particle);
    weights.push_back(1.0);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]); // Set gaussian distribution for x position 0 with std_x
  normal_distribution<double> dist_y(0, std_pos[1]); // Set gaussian distribution for y position 0 with std_y
  normal_distribution<double> dist_theta(0, std_pos[2]); // Set gaussian distribution for theta position 0 with std_theta
  
  for (int id = 0; id < num_particles; ++id) {
    if (fabs(yaw_rate) < 0.000001) {
      particles[id].x += velocity * delta_t * cos(particles[id].theta) + dist_x(gen);
      particles[id].y += velocity * delta_t * sin(particles[id].theta) + dist_y(gen);
      particles[id].theta += dist_theta(gen);
    } 
    else {
      particles[id].x += velocity / yaw_rate * (sin(particles[id].theta + yaw_rate * delta_t) - sin(particles[id].theta)) + dist_x(gen);
      particles[id].y += velocity / yaw_rate * (cos(particles[id].theta) - cos(particles[id].theta + yaw_rate * delta_t)) + dist_y(gen);
      particles[id].theta += yaw_rate * delta_t + dist_theta(gen);
    }
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  // Search all possible pairs for nearest neighbor
  for (int obs_id = 0; obs_id < observations.size(); ++obs_id) {
    LandmarkObs obs = observations[obs_id];
    double min_dist = -1; // Set min_dist to invalid integer
    int min_dist_id = -1; // Set min_dist_id to invalid integer
    
    for (int pred_id = 0; pred_id < predicted.size(); ++pred_id) {
      LandmarkObs pred = predicted[pred_id];
      
      double curr_dist = dist(obs.x, obs.y, pred.x, pred.y);
      if (min_dist == -1 || curr_dist < min_dist) {
        min_dist = curr_dist;
        min_dist_id = pred.id;
      }
    }
    
    observations[obs_id].id = min_dist_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  // For each particle
  for (int id = 0; id < num_particles; ++id) {
    std::vector<LandmarkObs> transformed_observations;
    
    // Transform observations from particle coordinates to map coordinates
    for (int obs_id = 0; obs_id < observations.size(); ++obs_id) {
      LandmarkObs obs = observations[obs_id];

      double x_map = (cos(particles[id].theta) * obs.x) - (sin(particles[id].theta) * obs.y) + particles[id].x;
      double y_map = (sin(particles[id].theta) * obs.x) + (cos(particles[id].theta) * obs.y) + particles[id].y;
      
      LandmarkObs tobs = {obs.id, x_map, y_map};
      
      transformed_observations.push_back(tobs);
    }
    
    // Select only landmarks which are possible to detect from the particle's frame of reference
    std::vector<LandmarkObs> possible_landmarks;
    for (int m_id = 0; m_id < map_landmarks.landmark_list.size(); ++m_id) {
      double ml_x = (double) map_landmarks.landmark_list[m_id].x_f;
      double ml_y = (double) map_landmarks.landmark_list[m_id].y_f;
      
      if (dist(ml_x, ml_y, particles[id].x, particles[id].y) <= sensor_range) {
        LandmarkObs ml = {map_landmarks.landmark_list[m_id].id_i, ml_x, ml_y};
        possible_landmarks.push_back(ml);
      }
    }
    
    // Associate each measurement with landmark identifier
    dataAssociation(possible_landmarks, transformed_observations);
    
    // Calculate particle's weight
    double weight = 1.0;
    
    for (int tobs_id = 0; tobs_id < transformed_observations.size(); ++tobs_id) {
      LandmarkObs tobs = transformed_observations[tobs_id];
      
      for (int i = 0; i < possible_landmarks.size(); ++i) {
        if (possible_landmarks[i].id == tobs.id) {
          weight *= multiv_prob(std_landmark[0], std_landmark[1], tobs.x, tobs.y, possible_landmarks[i].x, possible_landmarks[i].y);
        }
      }
    }
    
    // Adjust particle weights
    particles[id].weight = weight;
    weights[id] = weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::default_random_engine gen;
  std::discrete_distribution<> d(weights.begin(), weights.end());
  std::vector<Particle> resampled_particles;
  
  for (int id = 0; id < num_particles; ++id) {
    resampled_particles.push_back(particles[d(gen)]);
  }
  
  particles = std::move(resampled_particles);
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
