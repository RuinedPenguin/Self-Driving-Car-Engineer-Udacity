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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  std::default_random_engine gen;
  
  for (int i = 0; i < num_particles; i++) {
	Particle p;
	p.id = i;
	p.x = dist_x(gen);
	p.y = dist_y(gen);
	p.theta = dist_theta(gen);
    
	p.weight = 1.0;
	  
	particles.push_back(p);
	weights.push_back(p.weight);
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


  for(auto& p: particles){
    if( fabs(yaw_rate) < 0.0001){   // theta_dot = 0
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);

    } else{  						//theta_dot != 0
      p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
      p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
      p.theta += yaw_rate * delta_t;
    }
      std::default_random_engine gen;
	  std::normal_distribution<double> dist_x(p.x, std_pos[0]);
	  std::normal_distribution<double> dist_y(p.y, std_pos[1]);
	  std::normal_distribution<double> dist_theta(p.theta, std_pos[2]);
    
      p.x = dist_x(gen);
      p.y = dist_y(gen);
      p.theta = dist_theta(gen);
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

  for(auto& ob: observations){
    double min_dist = std::numeric_limits<float>::max();

    for(const auto& pr: predicted){
      double d = dist(ob.x, ob.y, pr.x, pr.y);
      if (d < min_dist){
        min_dist = d;
        ob.id = pr.id;
      }
    }
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
    double w_normalizer = 0.0;
  
 	for(auto& p: particles){
      p.weight = 1.0;
      
      double cos_theta = cos(p.theta);
      double sin_theta = sin(p.theta);

    // collecting valid landmarks
      vector<LandmarkObs> predictions;
      
      for(const auto& l: map_landmarks.landmark_list){
        double d = dist(p.x, p.y, l.x_f, l.y_f);
      	if( d <= sensor_range)
        	predictions.push_back(LandmarkObs{l.id_i, l.x_f, l.y_f});
      }

    // transformation from p to map
      vector<LandmarkObs> transformed;
      
      for(const auto& o: observations){
      	LandmarkObs temp;
      	temp.x = o.x * cos_theta - o.y * sin_theta + p.x;
      	temp.y = o.x * sin_theta + o.y * cos_theta + p.y;
      	temp.id = o.id;
      
      	transformed.push_back(temp);
      }

    // landmark association
      dataAssociation(predictions, transformed);
      
      
	  for(const auto& t: transformed){
     	Map::single_landmark_s l = map_landmarks.landmark_list.at(t.id-1);
      	double x = pow(t.x - l.x_f, 2) / (2 * pow(std_landmark[0], 2));
      	double y = pow(t.y - l.y_f, 2) / (2 * pow(std_landmark[1], 2));
      	double w = exp(-(x + y)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      	p.weight *=  w;
      }
      
	  w_normalizer += p.weight;
      weights.push_back(p.weight);
  }
  
  
  for (auto& p: particles) {
    p.weight /= w_normalizer;
    weights.push_back(p.weight);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  	vector<Particle> p3;
    p3.resize(num_particles);
  
	std::default_random_engine gen;
	
	std::uniform_int_distribution<int> index(0, num_particles - 1);
  
	int current_index = index(gen);
	double beta = 0.0;
	double mw2 = 2.0 * *max_element(weights.begin(), weights.end());
	
	for (unsigned int i = 0; i < particles.size(); i++) {
      std::uniform_real_distribution<double> rand(0.0, mw2);
	  beta += rand(gen);

	  while (beta > weights[current_index]) {
        beta -= weights[current_index];
	    current_index = (current_index + 1) % num_particles;
	    }
	
      p3.push_back(particles[current_index]);
	}
	particles = p3;
  
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