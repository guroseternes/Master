/*
 * Config.h
 *
 *  Created on: Jan 23, 2015
 *      Author: guro
 */

#ifndef CONFIG_H_
#define CONFIG_H_

enum Algorithm{BRUTE_FORCE, NEWTON_BRUTE_FORCE, NEWTON_BISECTION};
enum Perm{PERM_CONSTANT, PERM_VARIATIONAL};
enum Formation{UTSIRA, JOHANSEN};

struct Configurations {
	Algorithm algorithm_solve_h;
	Perm perm_type;
	Formation formation_name;
	char* formation;
	char* formation_dir;
	int vertical_resolution;
	float total_time;
	float injection_time;
	float initial_time_step;
	float beta;
	int pressure_update_injection;
	int pressure_update_migration;

};

#endif /* CONFIG_H_ */
