/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::normal_distribution;
using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta,
                          const double std[]) {

  num_particles = 50;
  particles = vector<Particle>(num_particles);
  weights = vector<double>(num_particles);

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    double sampled_x, sampled_y, sampled_theta;

    sampled_x = dist_x(gen);
    sampled_y = dist_y(gen);
    sampled_theta = dist_theta(gen);

    Particle particle;
    particle.id = i;
    particle.x = sampled_x;
    particle.y = sampled_y;
    particle.theta = sampled_theta;
    particle.weight = 1;

    particles[i] = particle;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {

  for (int i = 0; i < num_particles; ++i) {
    Particle &p = particles[i];
    double x, y, theta;
    double change_in_yaw = yaw_rate * delta_t;

    if (abs(change_in_yaw) < 0.00001) {
      // Very low or changes in yaw rate would yield divide by zero or near zero
      x = p.x + velocity * delta_t * cos(p.theta);
      y = p.y + velocity * delta_t * sin(p.theta);
    } else {
      x = p.x +
          (velocity / yaw_rate) * (sin(p.theta + change_in_yaw) - sin(p.theta));
      y = p.y +
          (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + change_in_yaw));
    }
    theta = p.theta + change_in_yaw;

    normal_distribution<double> dist_x(x, std_pos[0]);
    normal_distribution<double> dist_y(y, std_pos[1]);
    normal_distribution<double> dist_theta(theta, std_pos[2]);

    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {

  for (auto p_idx = 0; p_idx < num_particles; p_idx++) {

    auto &p = particles[p_idx];
    double prob_acc = 1;

    auto associations = vector<int>(observations.size());
    auto sense_x = vector<double>(observations.size());
    auto sense_y = vector<double>(observations.size());

    for (int obs_idx = 0; obs_idx < observations.size(); obs_idx++) {
      auto &obs = observations[obs_idx];
      // Transform observation to map co3ordinates
      // TODO: Validate that the translation is correct
      const double cos_theta = cos(p.theta);
      const double sin_theta = sin(p.theta);
      const double obs_x_map = obs.x * cos_theta - obs.y * sin_theta + p.x;
      const double obs_y_map = obs.x * sin_theta + obs.y * cos_theta + p.y;

      // Find the nearest landmark
      // NOTE: This can be improved with a tree based search
      auto landmark = findNearestLandmark(map_landmarks, obs_x_map, obs_y_map);
      associations[obs_idx] = landmark.id_i;
      sense_x[obs_idx] = obs_x_map;
      sense_y[obs_idx] = obs_y_map;

      // Calculate likelihood of observing this landmark in this location
      const double likelihood = multivariate_gaussian_pdf(
          obs_x_map, obs_y_map, landmark.x_f, landmark.y_f, std_landmark[0],
          std_landmark[1]);
      prob_acc *= likelihood;
    }

    // Update association for debugging purposes
    SetAssociations(p, associations, sense_x, sense_y);

    weights[p_idx] = prob_acc;
  }
}

Map::single_landmark_s
ParticleFilter::findNearestLandmark(const Map &map_landmarks,
                                    const double obs_x_map,
                                    const double obs_y_map) const {
  unsigned long nearest_landmark_idx = 0;
  double nearest_landmark_distance = std::numeric_limits<double>::max();
  for (auto l_idx = 0; l_idx < map_landmarks.landmark_list.size(); l_idx++) {
    auto landmark = map_landmarks.landmark_list[l_idx];
    const double distance_obs_landmark =
        dist(obs_x_map, obs_y_map, landmark.x_f, landmark.y_f);

    if (distance_obs_landmark < nearest_landmark_distance) {
      // This is the closest match so far
      nearest_landmark_idx = l_idx;
      nearest_landmark_distance = distance_obs_landmark;
    }
  }
  return map_landmarks.landmark_list[nearest_landmark_idx];
}

void ParticleFilter::resample() {

  auto w_copy = weights;

  std::discrete_distribution<> d(w_copy.begin(), w_copy.end());

  vector<Particle> surviving_particles(num_particles);

  for (auto j = 0; j < num_particles; j++) {
    auto p_idx = d(gen);
    surviving_particles[j] = particles[p_idx];
  }

  particles = surviving_particles;

  //  is_initialized = false;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
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
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
