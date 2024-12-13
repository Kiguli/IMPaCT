#include "IMDP.h"
#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <string>
#include <nlopt.hpp>
#include <iomanip>
#include <AdaptiveCpp/sycl/sycl.hpp>
#include <chrono>
#include <armadillo>
#include <hdf5/serial/hdf5.h>

using namespace std;
using namespace arma;


///Save minimal target transition vector
void IMDP::saveMinTargetTransitionVector(){
    if (minTargetM.empty()){
        cout << "Min Target Transition Vector is empty, can't save file." << endl;
    }else{
        minTargetM.save("minttm.h5", hdf5_binary);
    }
}

///Load minimal target transition vector
void IMDP::loadMinTargetTransitionVector(string filename){
    bool ok = minTargetM.load(filename);
    if (ok == false){
        cout << "Issue loading minimum target transition Vector!" << endl;
    }
}

///Save minimal avoid transition vector
void IMDP::saveMinAvoidTransitionVector(){
    if (minAvoidM.empty()){
        cout << "Min Avoid Transition Vector is empty, can't save file." << endl;
    }else{
        minAvoidM.save("minatm.h5", hdf5_binary);
    }
}

///Load minimal avoid transition vector
void IMDP::loadMinAvoidTransitionVector(string filename){
    bool ok = minAvoidM.load(filename);
    if (ok == false){
        cout << "Issue loading minimum avoid transition Vector!" << endl;
    }
}

///Save minimal transition matrix
void IMDP::saveMinTransitionMatrix(){
    if (minTransitionM.empty()){
        cout << "Min Transition Matrix is empty, can't save file." << endl;
    }else{
        minTransitionM.save("mintm.h5", hdf5_binary);
    }
}

///Load minimal transition matrix
void IMDP::loadMinTransitionMatrix(string filename){
    bool ok = minTransitionM.load(filename);
    if (ok == false){
        cout << "Issue loading minimum transition matrix!" << endl;
    }
}

///Save maximal target transition vector
void IMDP::saveMaxTargetTransitionVector(){
    if (maxTargetM.empty()){
        cout << "Max Target Transition Vector is empty, can't save file." << endl;
    }else{
        maxTargetM.save("maxttm.h5", hdf5_binary);
    }
}

///Load maximal target transition vector
void IMDP::loadMaxTargetTransitionVector(string filename){
    bool ok = maxTargetM.load(filename);
    if (ok == false){
        cout << "Issue loading maximum target transition Vector!" << endl;
    }
}

///Save maximal avoid transition vector
void IMDP::saveMaxAvoidTransitionVector(){
    if (maxAvoidM.empty()){
        cout << "Max Avoid Transition Vector is empty, can't save file." << endl;
    }else{
        maxAvoidM.save("maxatm.h5", hdf5_binary);
    }
}

///Load maximal avoid transition vector
void IMDP::loadMaxAvoidTransitionVector(string filename){
    bool ok = maxAvoidM.load(filename);
    if (ok == false){
        cout << "Issue loading maximum avoid transition Vector!" << endl;
    }
}

///Save maximal transition matrix
void IMDP::saveMaxTransitionMatrix(){
    if (maxTransitionM.empty()){
        cout << "Max Transition Matrix is empty, can't save file." << endl;
    }else{
        maxTransitionM.save("maxtm.h5", hdf5_binary);
    }
}

///Load maximal transition matrix
void IMDP::loadMaxTransitionMatrix(string filename){
    bool ok = maxTransitionM.load(filename);
    if (ok == false){
        cout << "Issue loading maximum transition matrix!" << endl;
    }
}

///Save Controller
void IMDP::saveController(){
    if (controller.empty()){
        cout << "Controller is empty, can't save file." << endl;
    }else{
        controller.save("controller.h5", hdf5_binary);
    }
}

/// Load Controller
void IMDP::loadController(string filename){
    bool ok = controller.load(filename);
    if (ok == false){
        cout << "Issue loading controller!" << endl;
    }
}

/// Destructor
IMDP::~IMDP(){
}

/// Sorted Implementation of infinite horizon reachability
void IMDP::infiniteHorizonReachControllerSorted(bool IMDP_lower){
    auto start = chrono::steady_clock::now();
    cout << "Finding control policy for infinite horizon reach controller using sorted approach... " << endl;
    
    if (input_space_size == 0 && disturb_space_size == 0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size, 1, fill::zeros);

            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;

                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);

                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }

                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });

                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }

                //Get difference between max and min for incrementing values
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;

                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;

                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }

                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }

                            // maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }

                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }

                            //return final values
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;

                        });
                    });
                }
                queue.wait_and_throw();

                vec check0 = firstnew0;
                vec check1 = firstnew1;
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;

                max_diff = max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;

            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);

            cout << "Create reduced matrix where input is fixed." << endl;
                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempTTmin= minTargetM;
                tempTTmax= maxTargetM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;

            cout << "Matrix Fixed" << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;


                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);

                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }

                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });

                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }

                //Get difference between max and min for incrementing values
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;


                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;

                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }

                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                                s+= accdTT[i];
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }

                            //rest is avoid state transitions we don't need to calculate

                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = secondnew0;
                second1 = secondnew1;

                max_diff = max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            cout << endl;
            cout << "Upper bound found." << endl;

            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = first0;
            controller.col(dim_x + 1) = second1;
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                //Get difference between max and min for incrementing values
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                                s = s+accdTT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                vec check0 = firstnew0;
                vec check1 = firstnew1;
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                max_diff = max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempTTmin= minTargetM;
                tempTTmax= maxTargetM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;
                
            cout << "Matrix Fixed" << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                //Get difference between max and min for incrementing values
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            //maximize transitions to target
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = secondnew0;
                second1 = secondnew1;
                
                max_diff = max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = second0;
            controller.col(dim_x + 1) = first1;
        }
    }else if (disturb_space_size == 0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                //Get difference between max and min for incrementing values
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*input_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*input_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                firstnew1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(max(firstnew0,1));
                vec check1 = conv_to< colvec >::from(max(firstnew1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
                
                max_diff = max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempTTmin(i)= minTargetM(U_pos(i)*state_space_size+i);
                tempTTmax(i)= maxTargetM(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }
            
            cout << "Matrix Fixed" << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                //Get difference between max and min for incrementing values
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                                s+= accdTT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = secondnew0;
                second1 = secondnew1;
                
                max_diff = max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*input_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }
                            
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                                s = s+accdTT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*input_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                firstnew1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(max(firstnew0,1));
                vec check1 = conv_to< colvec >::from(max(firstnew1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
                
                max_diff = max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempTTmin(i)= minTargetM(U_pos(i)*state_space_size+i);
                tempTTmax(i)= maxTargetM(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }
            
            cout << "Matrix Fixed" << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            //maximize transitions to target
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = secondnew0;
                second1 = secondnew1;
                
                max_diff = max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = second0;
            controller.col(dim_x+dim_u + 1) = first1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }else if (input_space_size==0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                //Get difference between max and min for incrementing values
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size,disturb_space_size);
                firstnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(min(firstnew0,1));
                vec check1 = conv_to< colvec >::from(min(firstnew1,1));
                
                
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                max_diff = max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                //Get difference between max and min for incrementing values
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> buff1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> buff0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(min(secondnew0,1));
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                
                
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = check0;
                second1 = check1;
                
                max_diff = max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = first0;
            controller.col(dim_x + 1) = second1;
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size,disturb_space_size);
                firstnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(min(firstnew0,1));
                vec check1 = conv_to< colvec >::from(min(firstnew1,1));
                
                
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                max_diff = max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> buff1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> buff0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(min(secondnew0,1));
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                
                
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = check0;
                second1 = check1;
                
                max_diff = max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = second0;
            controller.col(dim_x + 1) = first1;
        }
    }
    else{
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state0(input_space_size*state_space_size, 1, fill::zeros);
            mat input_and_state1(input_space_size*state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            
            cout << "first loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                //Get difference between max and min for incrementing values
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size*input_space_size,disturb_space_size);
                firstnew1.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state0 = min(firstnew0,1);
                input_and_state1 = min(firstnew1,1);
                
                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                input_and_state1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(max(input_and_state0,1));
                vec check1 = conv_to< colvec >::from(max(input_and_state1,1));
                
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
                
                max_diff = max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmin(j*state_space_size+i)= minTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmax(j*state_space_size+i)= maxTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }
            
            cout << "Matrix Fixed" << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                //Get difference between max and min for incrementing values
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                                s+= accdTT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(min(secondnew0,1));
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = check0;
                second1 = check1;
                
                max_diff = max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state0(input_space_size*state_space_size, 1, fill::zeros);
            mat input_and_state1(input_space_size*state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];
                            }
                            
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                                s = s+accdTT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size*input_space_size,disturb_space_size);
                firstnew1.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state0 = min(firstnew0,1);
                input_and_state1 = min(firstnew1,1);
                
                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                input_and_state1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(max(input_and_state0,1));
                vec check1 = conv_to< colvec >::from(max(input_and_state1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
                
                max_diff = max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmin(j*state_space_size+i)= minTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmax(j*state_space_size+i)= maxTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }
            
            cout << "Matrix Fixed" << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            //maximize transitions to target
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(min(secondnew0,1));
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = check0;
                second1 = check1;
                
                max_diff = max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = second0;
            controller.col(dim_x+dim_u + 1) = first1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

// Sorted Method for finite horizon reachability
void IMDP::finiteHorizonReachControllerSorted(bool IMDP_lower, size_t timeHorizon){
    auto start = chrono::steady_clock::now();
    cout << "Finding control policy for finite horizon reach controller using sorted approach... " << endl;
    
    if (input_space_size == 0 && disturb_space_size == 0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size, 1, fill::zeros);
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k < timeHorizon) {
                cout << "." << flush;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first0);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }
                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                queue.wait_and_throw();
                k++;
                first0 = firstnew0;
                
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            k = 0;
            
            cout << "Create reduced matrix where input is fixed." << endl;
            
                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempTTmin= minTargetM;
                tempTTmax= maxTargetM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;

            
            
            cout << "Matrix Fixed" << endl;
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second0);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            
                            double temp0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                                
                            }
                            
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                s+= accdTT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second0 = secondnew0;
                
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size, 1, fill::zeros);
            
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k < timeHorizon) {
            cout << "." << flush;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first0);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;

                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), 0);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), 0);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), 0);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            
                            // maximize transitions to target set
                            
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                s = s+accdTT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            cdfAccessor0[i] =  temp0;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                k++;
                first0 = firstnew0;
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            k = 0;
            
            cout << "Create reduced matrix where input is fixed." << endl;
            
                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempTTmin= minTargetM;
                tempTTmax= maxTargetM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;

            cout << "Matrix Fixed" << endl;
            while (k < timeHorizon) {
                cout << "." << flush;
                
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second0);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            //maximize transitions to target
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second0 = secondnew0;
                
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
        }
    }else if (input_space_size==0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k < timeHorizon) {
                cout << "." << flush;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first0);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;

                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);

                            }else{
                                temp0 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size,disturb_space_size);
                first0 = conv_to< colvec >::from(min(firstnew0,1)); 
                k++;
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            k = 0;
            cout << "second loop iterations: " << endl;
            while (k < timeHorizon) {
                cout << "." << flush;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second0);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;

                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> buff0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                second0 = conv_to< colvec >::from(min(secondnew0,1));
                
                k++;
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = first0;
            controller.col(dim_x + 1) = second0;
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k<timeHorizon) {
                cout << "." << flush;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first0);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;

                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }

                            cdfAccessor0[i] =  temp0;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size,disturb_space_size);
                first0 = conv_to< colvec >::from(min(firstnew0,1));
                k++;
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            while (k<timeHorizon) {
                cout << "." << flush;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second0);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> buff0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                second0 = conv_to< colvec >::from(min(secondnew0,1));
                k++;
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = second0;
            controller.col(dim_x + 1) = first0;
        }
    }
    
    else if (disturb_space_size == 0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k < timeHorizon) {
            cout << "." << flush;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first0);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;

                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                //TODO: throw an error here.
                                temp0 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            
                        });
                    });
                }
                queue.wait_and_throw();
               
                
                /*Resize to maximise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                first0 = conv_to< colvec >::from(max(firstnew0,1));
                k++;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            k=0;
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempTTmin(i)= minTargetM(U_pos(i)*state_space_size+i);
                tempTTmax(i)= maxTargetM(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }
            
           
            cout << "Matrix Fixed" << endl;
            while (k < timeHorizon) {
                cout << "." << flush;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second0);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                //Get difference between max and min for incrementing values
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;


                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                s+= accdTT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second0 = secondnew0;
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k < timeHorizon) {
                cout << "." << flush;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first0);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                //Get difference between max and min for incrementing values
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;

                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }
                            
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                s = s+accdTT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            cdfAccessor0[i] =  temp0;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                first0 = conv_to< colvec >::from(max(firstnew0,1));
                k++;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
                
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            k=0;
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempTTmin(i)= minTargetM(U_pos(i)*state_space_size+i);
                tempTTmax(i)= maxTargetM(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }
            
            
            
            cout << "Matrix Fixed" << endl;
            while (k<timeHorizon) {
                
                cout << "." << flush;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second0);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[col];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            //maximize transitions to target
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second0 = secondnew0;
                
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }else{
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state0(input_space_size*state_space_size, 1, fill::zeros);
            
            size_t k=0;
            cout << "first loop iterations: " << endl;
            while (k < timeHorizon) {
                cout << "." << flush;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first0);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;


                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state0 = min(firstnew0,1);
                
                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                first0 = conv_to< colvec >::from(max(input_and_state0,1));
                
                k++;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
                
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
            k=0;
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmin(j*state_space_size+i)= minTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmax(j*state_space_size+i)= maxTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }
           
            
            cout << "Matrix Fixed" << endl;
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second0);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                s+= accdTT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                second0 = conv_to< colvec >::from(min(secondnew0,1));
                
                k++;
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state0(input_space_size*state_space_size, 1, fill::zeros);
            
            size_t k=0;
            cout << "first loop iterations: " << endl;
            while (k < timeHorizon) {
            cout << "." << endl; 
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first0);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                
                TargetM = minTargetM;
                AvoidM = minAvoidM;
                TransitionM = minTransitionM;
                //}

                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), TargetM.n_rows);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), AvoidM.n_rows);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), TransitionM.n_rows * TransitionM.n_cols);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        auto accTargetM = bufTargetM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accAvoidM = bufAvoidM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accTransitionM = bufTransitionM.get_access<sycl::access::mode::read_write>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];
                                
                            }
                            
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                accTargetM[i] += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                accTargetM[i] += accdTT[i];
                                s = s+accdTT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    accTransitionM[(val*state_space_size*input_space_size*disturb_space_size) +i] += (1.0-s);
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf0[val];
                                    accTransitionM[(val*state_space_size*input_space_size*disturb_space_size) +i] += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            accAvoidM[i] = (1.0-s);
                            cdfAccessor0[i] =  temp0;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state0 = min(firstnew0,1);
                
                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                first0 = conv_to< colvec >::from(max(input_and_state0,1));
                k++;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
            
            k=0;
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmin(j*state_space_size+i)= minTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmax(j*state_space_size+i)= maxTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }
            
            cout << "Matrix Fixed" << endl;
            while (k<timeHorizon) {
                cout << "." << flush;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second0);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            //maximize transitions to target
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                second0 = conv_to< colvec >::from(min(secondnew0,1));
                k++;
                
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}


// Sorted Method for infinite horizon safety
void IMDP::infiniteHorizonSafeControllerSorted(bool IMDP_lower){
    auto start = chrono::steady_clock::now();
    cout << "Finding control policy for infinite horizon safe controller using sorted approach... " << endl;
    
    if (input_space_size == 0 && disturb_space_size == 0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                                
                            }
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                vec check0 = firstnew0;
                vec check1 = firstnew1;
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                max_diff = max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;
            
            cout << "Matrix Fixed" << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                                s+= accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = secondnew0;
                second1 = secondnew1;
                
                max_diff = max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = ones(state_space_size)-first1;
            controller.col(dim_x + 1) = ones(state_space_size)-second1;
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                                s = s+accdAT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                vec check0 = firstnew0;
                vec check1 = firstnew1;
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                max_diff = max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;
            
            cout << "Matrix Fixed" << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = secondnew0;
                second1 = secondnew1;
                
                max_diff = max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = ones(state_space_size)-second1;
            controller.col(dim_x + 1) = ones(state_space_size)-first1;
        }
    }else if (disturb_space_size == 0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*input_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*input_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                firstnew1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(min(firstnew0,1));
                vec check1 = conv_to< colvec >::from(min(firstnew1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).min(U_pos[i]);
                }
                
                max_diff = max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }
            
            
            cout << "Matrix Fixed" << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                                s+= accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = secondnew0;
                second1 = secondnew1;
                
                max_diff = max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = ones(state_space_size)-first1;
            controller.col(dim_x+dim_u + 1) = ones(state_space_size)-second1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*input_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }
                            
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                                s = s+accdAT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*input_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                firstnew1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(min(firstnew0,1));
                vec check1 = conv_to< colvec >::from(min(firstnew1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).min(U_pos[i]);
                }
                
                max_diff = max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }
            
            cout << "Matrix Fixed" << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                         
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = secondnew0;
                second1 = secondnew1;
                
                max_diff = max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = ones(state_space_size)-second1;
            controller.col(dim_x+dim_u + 1) = ones(state_space_size)-first1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }else if (input_space_size==0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size,disturb_space_size);
                firstnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(max(firstnew0,1));
                vec check1 = conv_to< colvec >::from(max(firstnew1,1));
                
                
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                max_diff = max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> buff1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> buff0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(max(secondnew0,1));
                vec check1 = conv_to< colvec >::from(max(secondnew1,1));
                
                
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = check0;
                second1 = check1;
                
                max_diff = max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = ones(state_space_size)-first1;
            controller.col(dim_x + 1) = ones(state_space_size)-second1;
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size,disturb_space_size);
                firstnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(max(firstnew0,1));
                vec check1 = conv_to< colvec >::from(max(firstnew1,1));
                
                
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                max_diff = max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> buff1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> buff0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(max(secondnew0,1));
                vec check1 = conv_to< colvec >::from(max(secondnew1,1));
                
                
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = check0;
                second1 = check1;
                
                max_diff = max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = ones(state_space_size)-second1;
            controller.col(dim_x + 1) = ones(state_space_size)-first1;
        }
    }
    else{
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state0(input_space_size*state_space_size, 1, fill::zeros);
            mat input_and_state1(input_space_size*state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];
                            }
                            
                            
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size*input_space_size,disturb_space_size);
                firstnew1.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state0 = max(firstnew0,1);
                input_and_state1 = max(firstnew1,1);
                
                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                input_and_state1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(min(input_and_state0,1));
                vec check1 = conv_to< colvec >::from(min(input_and_state1,1));
                
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).min(U_pos[i]);
                }
                
                max_diff = max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }
           
            cout << "Matrix Fixed" << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                                s+= accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(max(secondnew0,1));
                vec check1 = conv_to< colvec >::from(max(secondnew1,1));
                
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = check0;
                second1 = check1;
                
                max_diff = max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = ones(state_space_size)-first1;
            controller.col(dim_x+dim_u + 1) = ones(state_space_size)-second1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state0(input_space_size*state_space_size, 1, fill::zeros);
            mat input_and_state1(input_space_size*state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    //sycl::buffer<double> bufS(s.memptr(),s.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];
                            }
                            
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                                s = s+accdAT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size*input_space_size,disturb_space_size);
                firstnew1.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state0 = max(firstnew0,1);
                input_and_state1 = max(firstnew1,1);
                
                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                input_and_state1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(min(input_and_state0,1));
                vec check1 = conv_to< colvec >::from(min(input_and_state1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).min(U_pos[i]);
                }
                
                max_diff = max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }
            
            cout << "Matrix Fixed" << endl;
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;

                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            //maximize transitions to target
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(max(secondnew0,1));
                vec check1 = conv_to< colvec >::from(max(secondnew1,1));
                
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = check0;
                second1 = check1;
                
                max_diff = max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = ones(state_space_size)-second1;
            controller.col(dim_x+dim_u + 1) = ones(state_space_size)-first1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

//Sorted Method for finite horizon safety
void IMDP::finiteHorizonSafeControllerSorted(bool IMDP_lower, size_t timeHorizon){
    auto start = chrono::steady_clock::now();
    cout << "Finding control policy for finite horizon safe controller using sorted approach... " << endl;
    
    if (input_space_size == 0 && disturb_space_size == 0){
        if (!IMDP_lower){
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size, 1, fill::zeros);

            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    //sycl::buffer<double> bufS(s.memptr(),s.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                vec check1 = firstnew1;
                k++;
                first1 = check1;
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;
            
            cout << "Matrix Fixed" << endl;
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second1 = secondnew1;
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = first1;
            controller.col(dim_x + 1) = second1;
        }
        else{
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                vec check1 = firstnew1;
                k++;
                first1 = check1;
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;
            
            cout << "Matrix Fixed" << endl;
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second1 = secondnew1;
                
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = second1;
            controller.col(dim_x + 1) = first1;
        }
    }else if (disturb_space_size == 0){
        if (!IMDP_lower){
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*input_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }
                            
                            //transitions to avoid
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*input_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over input*/
                firstnew1.reshape(state_space_size, input_space_size);
                vec check1 = conv_to< colvec >::from(max(firstnew1,1));
                k++;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew1.row(i).max(U_pos[i]);
                }
                
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }
            
            
            cout << "Matrix Fixed" << endl;
            while (k <timeHorizon) {
                cout << "." << flush;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second1 = secondnew1;
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first1;
            controller.col(dim_x+dim_u + 1) = second1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    //sycl::buffer<double> bufS(s.memptr(),s.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*input_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }
                            
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*input_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over input*/
                firstnew1.reshape(state_space_size, input_space_size);
                vec check1 = conv_to< colvec >::from(max(firstnew1,1));
                k++;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew1.row(i).max(U_pos[i]);
                }
                
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }
            
            cout << "Matrix Fixed" << endl;
            while (k<timeHorizon) {
                cout << "." << flush;
                
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                         
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second1 = secondnew1;
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first1;
            controller.col(dim_x+dim_u + 1) = second1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }else if (input_space_size==0){
        if (!IMDP_lower){
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k<timeHorizon) {
                cout << "." << flush;
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    //sycl::buffer<double> bufS(s.memptr(),s.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        //auto accs = bufS.get_access<sycl::access::mode::read_write>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew1.reshape(state_space_size,disturb_space_size);
                vec check1 = conv_to< colvec >::from(min(firstnew1,1));
                
                k++;
                first1 = check1;
                
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> buff1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[val];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                k++;
                second1 = check1;
                
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = first1;
            controller.col(dim_x + 1) = second1;
        }
        else{
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k<timeHorizon) {
                cout << "." << flush;
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    //sycl::buffer<double> bufS(s.memptr(),s.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew1.reshape(state_space_size,disturb_space_size);
                vec check1 = conv_to< colvec >::from(min(firstnew1,1));
                k++;
                first1 = check1;
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            while (k<timeHorizon) {
                cout << "." << flush;
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> buff1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                k++;
                second1 = check1;
                
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = second1;
            controller.col(dim_x + 1) = first1;
        }
    }
    else{
        if (!IMDP_lower){
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state1(input_space_size*state_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k<timeHorizon) {
                cout << "." << flush;
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    //sycl::buffer<double> bufS(s.memptr(),s.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];
                            }

                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew1.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state1 = min(firstnew1,1);
                
                /*Resize to maximise over input*/
                input_and_state1.reshape(state_space_size, input_space_size);
                vec check1 = conv_to< colvec >::from(max(input_and_state1,1));
                k++;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew1.row(i).max(U_pos[i]);
                }
                
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }
           
            cout << "Matrix Fixed" << endl;
            while (k<timeHorizon) {
                cout << "." << flush;
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                k++;
                second1 = check1;
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first1;
            controller.col(dim_x+dim_u + 1) = second1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state1(input_space_size*state_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k<timeHorizon) {
                cout << "." << flush;
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    //sycl::buffer<double> bufS(s.memptr(),s.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];
                            }
                            
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew1.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state1 = min(firstnew1,1);
                
                /*Resize to maximise over input*/
                input_and_state1.reshape(state_space_size, input_space_size);
                vec check1 = conv_to< colvec >::from(max(input_and_state1,1));
                k++;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew1.row(i).max(U_pos[i]);
                }
                
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }
            
            cout << "Matrix Fixed" << endl;
            while (k<timeHorizon) {
                cout << "." << flush;
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second1);
                
                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }
                
                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
                
                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }
                
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
                
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                k++;
                second1 = check1;
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = second1;
            controller.col(dim_x+dim_u + 1) = first1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

/*Collect MDP that is synthesized for counter-factual analysis*/

// Sorted Method for finite horizon safety
void IMDP::finiteHorizonReachControllerSortedStoreMDP(bool IMDP_lower, size_t timeHorizon){
    auto start = chrono::steady_clock::now();
    cout << "Finding control policy for finite horizon reach controller using sorted approach... " << endl;

    if (input_space_size == 0 && disturb_space_size == 0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size, 1, fill::zeros);
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k < timeHorizon) {
                cout << "." << flush;

                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first0);

                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }

                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });

                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }

                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;


                TargetM = minTargetM;
                AvoidM = minAvoidM;
                TransitionM = minTransitionM;

                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), 0);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), 0);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), 0);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //only used if buffer is bigger than zero
                        auto accTargetM = bufTargetM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accAvoidM = bufAvoidM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accTransitionM = bufTransitionM.get_access<sycl::access::mode::read_write>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }

                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                //if (storeMDP==true) {
                                accAvoidM[i] += (1.0-s);
                                //}
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                                //if (storeMDP==true) {
                                accAvoidM[i] += accdAT[i];
                                //}
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    //if (storeMDP==true) {
                                    accTransitionM[(val*state_space_size) +i] += (1.0-s);
                                    //}
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                    //if (storeMDP==true) {
                                    accTransitionM[(val*state_space_size) +i] += accdT[(val*state_space_size) +i];
                                    //}
                                }
                            }

                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                //if (storeMDP==true) {
                                accTargetM[i] += (1.0-s);
                                //}
                            }else{
                                temp0 += accdTT[i];
                                //if (storeMDP==true) {
                                accTargetM[i] += accdTT[i];
                                //}
                            }
                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                queue.wait_and_throw();
                k++;
                first0 = firstnew0;

            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;

            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);

            k = 0;

            cout << "Create reduced matrix where input is fixed." << endl;

                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempTTmin= minTargetM;
                tempTTmax= maxTargetM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;



            cout << "Matrix Fixed" << endl;
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second0);

                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }

                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });

                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }

                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;


                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;

                            double temp0;
                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size) +i];

                            }

                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                s+= accdTT[i];
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }

                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second0 = secondnew0;

            }
            cout << endl;
            cout << "Upper bound found." << endl;

            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size, 1, fill::zeros);


            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k < timeHorizon) {
            cout << "." << flush;

                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first0);

                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }

                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });

                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }

                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;

                //if (storeMDP==true) {
                TargetM = minTargetM;
                AvoidM = minAvoidM;
                TransitionM = minTransitionM;
                //}

                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), 0);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), 0);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), 0);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        //only used if buffer is bigger than zero
                       auto accTargetM = bufTargetM.get_access<sycl::access::mode::read_write>(cgh);
                       auto accAvoidM = bufAvoidM.get_access<sycl::access::mode::read_write>(cgh);
                       auto accTransitionM = bufTransitionM.get_access<sycl::access::mode::read_write>(cgh);
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];


                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }


                            // maximize transitions to target set

                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                //if (storeMDP==true) {
                                accTargetM[i] += (1.0-s);
                                //}
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                //if (storeMDP==true) {
                                accTargetM[i] += accdTT[i];
                                //}
                                s = s+accdTT[i];
                            }

                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    //if (storeMDP==true) {
                                    accTransitionM[(val*state_space_size) +i] += (1.0-s);
                                    //}
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accf0[val];
                                    //if (storeMDP==true) {
                                    accTransitionM[(val*state_space_size) +i] += accdT[(val*state_space_size) +i];
                                    //}
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }

                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            //if (storeMDP==true) {
                            accAvoidM[i] += (1.0-s);
                                //}
                            cdfAccessor0[i] =  temp0;

                        });
                    });
                }
                queue.wait_and_throw();
                k++;
                first0 = firstnew0;
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;


            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);

            k = 0;

            cout << "Create reduced matrix where input is fixed." << endl;

                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempTTmin= minTargetM;
                tempTTmax= maxTargetM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;

            cout << "Matrix Fixed" << endl;
            while (k < timeHorizon) {
                cout << "." << flush;


                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second0);

                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }

                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });

                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }

                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;


                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;
                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }

                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }

                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }

                            //maximize transitions to target
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }

                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second0 = secondnew0;

            }
            cout << endl;
            cout << "Upper bound found." << endl;

            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
        }
    }else if (input_space_size==0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);

            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k < timeHorizon) {
                cout << "." << flush;

                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first0);

                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }

                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });

                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }

                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;

                TargetM = minTargetM;
                AvoidM = minAvoidM;
                TransitionM = minTransitionM;

                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), TargetM.n_rows);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), AvoidM.n_rows);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), TransitionM.n_rows * TransitionM.n_cols);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        auto accTargetM = bufTargetM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accAvoidM = bufAvoidM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accTransitionM = bufTransitionM.get_access<sycl::access::mode::read_write>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];


                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }

                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                                accAvoidM[i] += (1.0-s);
                            }else{
                                s = s+accdAT[i];
                                accAvoidM[i] += accdAT[i];
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    accTransitionM[(val*state_space_size*disturb_space_size) +i] += (1.0-s);
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    accTransitionM[(val*state_space_size*disturb_space_size) +i] += accdT[(val*state_space_size*disturb_space_size) +i];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }

                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                accTargetM[i] += (1.0-s);

                            }else{
                                temp0 += accdTT[i];
                                accTargetM[i] += accdTT[i];
                            }

                            cdfAccessor0[i] =  temp0;

                        });
                    });
                }
                queue.wait_and_throw();

                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size,disturb_space_size);
                first0 = conv_to< colvec >::from(min(firstnew0,1));
                k++;
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;

            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            k = 0;
            cout << "second loop iterations: " << endl;
            while (k < timeHorizon) {
                cout << "." << flush;

                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second0);

                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }

                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });

                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }

                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;

                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> buff0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];


                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }

                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }

                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }

                            cdfAccessor0[i] =  temp0;

                        });
                    });
                }
                queue.wait_and_throw();

                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                second0 = conv_to< colvec >::from(min(secondnew0,1));

                k++;
            }
            cout << endl;
            cout << "Upper bound found." << endl;

            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = first0;
            controller.col(dim_x + 1) = second0;
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);

            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k<timeHorizon) {
                cout << "." << flush;

                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first0);

                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }

                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });

                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }

                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;

                TargetM = minTargetM;
                AvoidM = minAvoidM;
                TransitionM = minTransitionM;

                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), TargetM.n_rows);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), AvoidM.n_rows);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), TransitionM.n_rows * TransitionM.n_cols);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        auto accTargetM = bufTargetM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accAvoidM = bufAvoidM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accTransitionM = bufTransitionM.get_access<sycl::access::mode::read_write>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }

                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                accTargetM[i] += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                accTargetM[i] += accdTT[i];
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    accTransitionM[(val*state_space_size*disturb_space_size) +i] += (1.0-s);
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    accTransitionM[(val*state_space_size*disturb_space_size) +i] += accdT[(val*state_space_size*disturb_space_size) +i];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }

                            accAvoidM[i] += (1.0-s);
                            cdfAccessor0[i] =  temp0;

                        });
                    });
                }
                queue.wait_and_throw();


                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size,disturb_space_size);
                first0 = conv_to< colvec >::from(min(firstnew0,1));
                k++;
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;

            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            while (k<timeHorizon) {
                cout << "." << flush;

                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second0);

                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }

                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });

                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }

                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;

                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> buff0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }

                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }

                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }

                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }

                            cdfAccessor0[i] =  temp0;

                        });
                    });
                }
                queue.wait_and_throw();

                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                second0 = conv_to< colvec >::from(min(secondnew0,1));
                k++;
            }
            cout << endl;
            cout << "Upper bound found." << endl;

            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = second0;
            controller.col(dim_x + 1) = first0;
        }
    }

    else if (disturb_space_size == 0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);

            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k < timeHorizon) {
            cout << "." << flush;

                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first0);

                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }

                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });

                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }

                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;

                //if (storeMDP==true) {
                TargetM = minTargetM;
                AvoidM = minAvoidM;
                TransitionM = minTransitionM;
                //}

                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), TargetM.n_rows);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), AvoidM.n_rows);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), TransitionM.n_rows * TransitionM.n_cols);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        auto accTargetM = bufTargetM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accAvoidM = bufAvoidM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accTransitionM = bufTransitionM.get_access<sycl::access::mode::read_write>(cgh);
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];


                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }

                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                                //if (storeMDP==true) {
                                accAvoidM[i] += (1.0-s);
                                //}
                            }else{
                                s = s+accdAT[i];
                                //if (storeMDP==true) {
                                accAvoidM[i] += accdAT[i];
                                //}
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    //if (storeMDP==true) {
                                    accTransitionM[(val*state_space_size*input_space_size) +i] += (1.0-s);
                                    //}
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size) +i]*accf0[val];
                                    //if (storeMDP==true) {
                                    accTransitionM[(val*state_space_size*input_space_size) +i] += accdT[(val*state_space_size*input_space_size) +i];
                                    //}
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }

                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                //if (storeMDP==true) {
                                accTargetM[i] += (1.0-s);
                                //}
                            }else{
                                //TODO: throw an error here.
                                temp0 += accdTT[i];
                                //if (storeMDP==true) {
                                accTargetM[i] += accdTT[i];
                                //}
                            }

                            cdfAccessor0[i] =  temp0;

                        });
                    });
                }
                queue.wait_and_throw();


                /*Resize to maximise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                first0 = conv_to< colvec >::from(max(firstnew0,1));
                k++;

                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;


            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);

            k=0;

            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempTTmin(i)= minTargetM(U_pos(i)*state_space_size+i);
                tempTTmax(i)= maxTargetM(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }


            cout << "Matrix Fixed" << endl;
            while (k < timeHorizon) {
                cout << "." << flush;

                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second0);

                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }

                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });

                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }

                //Get difference between max and min for incrementing values
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;


                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }

                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                s+= accdTT[i];
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }

                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second0 = secondnew0;
            }
            cout << endl;
            cout << "Upper bound found." << endl;

            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);

            size_t k = 0;
            cout << "first loop iterations: " << endl;
            while (k < timeHorizon) {
                cout << "." << flush;

                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first0);

                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }

                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });

                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }

                //Get difference between max and min for incrementing values
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;

                TargetM = minTargetM;
                AvoidM = minAvoidM;
                TransitionM = minTransitionM;
                //}

                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), TargetM.n_rows);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), AvoidM.n_rows);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), TransitionM.n_rows * TransitionM.n_cols);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        auto accTargetM = bufTargetM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accAvoidM = bufAvoidM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accTransitionM = bufTransitionM.get_access<sycl::access::mode::read_write>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }


                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                accTargetM[i] += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                accTargetM[i] += accdTT[i];
                                s = s+accdTT[i];
                            }

                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    accTransitionM[(val*state_space_size*input_space_size) +i] += (1.0-s);
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size) +i]*accf0[val];
                                    accTransitionM[(val*state_space_size*input_space_size) +i] += accdT[(val*state_space_size*input_space_size) +i];
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }

                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            accAvoidM[i] += (1.0-s);
                            cdfAccessor0[i] =  temp0;

                        });
                    });
                }
                queue.wait_and_throw();

                /*Resize to maximise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                first0 = conv_to< colvec >::from(max(firstnew0,1));
                k++;

                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }

            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;

            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);

            k=0;
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempTTmin(i)= minTargetM(U_pos(i)*state_space_size+i);
                tempTTmax(i)= maxTargetM(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }



            cout << "Matrix Fixed" << endl;
            while (k<timeHorizon) {

                cout << "." << flush;

                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second0);

                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }

                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });

                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }

                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;


                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }

                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }

                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[col];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }

                            //maximize transitions to target
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }

                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second0 = secondnew0;

            }
            cout << endl;
            cout << "Upper bound found." << endl;

            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }else{
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state0(input_space_size*state_space_size, 1, fill::zeros);

            size_t k=0;
            cout << "first loop iterations: " << endl;
            while (k < timeHorizon) {
                cout << "." << flush;

                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first0);

                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }

                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });

                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }

                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;

                TargetM = minTargetM;
                AvoidM = minAvoidM;
                TransitionM = minTransitionM;
                //}

                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), TargetM.n_rows);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), AvoidM.n_rows);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), TransitionM.n_rows * TransitionM.n_cols);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        auto accTargetM = bufTargetM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accAvoidM = bufAvoidM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accTransitionM = bufTransitionM.get_access<sycl::access::mode::read_write>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];


                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];
                            }

                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                accAvoidM[i] += (1.0-s);
                                s = 1.0;
                            }else{
                                accAvoidM[i] += accdAT[i];
                                s = s+accdAT[i];
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    accTransitionM[(val*state_space_size*input_space_size*disturb_space_size) +i] += (1.0-s);
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf0[val];
                                    accTransitionM[(val*state_space_size*input_space_size*disturb_space_size) +i] += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }

                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                accTargetM[i] += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                accTargetM[i] += accdTT[i];
                            }

                            cdfAccessor0[i] =  temp0;

                        });
                    });
                }
                queue.wait_and_throw();


                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state0 = min(firstnew0,1);

                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                first0 = conv_to< colvec >::from(max(input_and_state0,1));

                k++;

                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }

            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;

            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
            k=0;
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmin(j*state_space_size+i)= minTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmax(j*state_space_size+i)= maxTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }


            cout << "Matrix Fixed" << endl;
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second0);

                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }

                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });

                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }

                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;


                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }

                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                s+= accdTT[i];
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }

                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();

                /*Resize to maximise over disturbance - best case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                second0 = conv_to< colvec >::from(min(secondnew0,1));

                k++;
            }
            cout << endl;
            cout << "Upper bound found." << endl;

            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state0(input_space_size*state_space_size, 1, fill::zeros);

            size_t k=0;
            cout << "first loop iterations: " << endl;
            while (k < timeHorizon) {
            cout << "." << endl;

                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(first0);

                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }

                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });

                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }

                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;

                TargetM = minTargetM;
                AvoidM = minAvoidM;
                TransitionM = minTransitionM;
                //}

                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), TargetM.n_rows);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), AvoidM.n_rows);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), TransitionM.n_rows * TransitionM.n_cols);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        auto accTargetM = bufTargetM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accAvoidM = bufAvoidM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accTransitionM = bufTransitionM.get_access<sycl::access::mode::read_write>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];

                            }


                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                accTargetM[i] += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                accTargetM[i] += accdTT[i];
                                s = s+accdTT[i];
                            }

                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    accTransitionM[(val*state_space_size*input_space_size*disturb_space_size) +i] += (1.0-s);
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf0[val];
                                    accTransitionM[(val*state_space_size*input_space_size*disturb_space_size) +i] += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }

                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            accAvoidM[i] = (1.0-s);
                            cdfAccessor0[i] =  temp0;

                        });
                    });
                }
                queue.wait_and_throw();

                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state0 = min(firstnew0,1);

                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                first0 = conv_to< colvec >::from(max(input_and_state0,1));
                k++;

                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;

            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);

            k=0;
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmin(j*state_space_size+i)= minTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmax(j*state_space_size+i)= maxTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }

            cout << "Matrix Fixed" << endl;
            while (k<timeHorizon) {
                cout << "." << flush;

                std::vector<double> original_values = arma::conv_to < std::vector<double> >::from(second0);

                // Create a vector of pairs containing the original values and their indices
                std::vector<std::pair<int, double>> indexed_values;
                for (int i = 0; i < original_values.size(); ++i) {
                    indexed_values.push_back(std::make_pair(i, original_values[i]));
                }

                // Sort the vector based on the values
                std::sort(indexed_values.begin(), indexed_values.end(),
                          [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });

                // Extract the sorted indices
                std::vector<int> sorted_indices;
                for (const auto& pair : indexed_values) {
                    sorted_indices.push_back(pair.first);
                }

                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;


                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }

                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }

                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }

                            //maximize transitions to target
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }

                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                second0 = conv_to< colvec >::from(min(secondnew0,1));
                k++;

            }
            cout << endl;
            cout << "Upper bound found." << endl;

            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}