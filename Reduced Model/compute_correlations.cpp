#include<iostream>
#include<cstdlib>
#include<vector>
#include<cmath>
#include<fstream>
#include<random>
#include<cstring>

#define SINGLE 0
#define SIGNALSWEEP 1
#define WEIGHTSWEEP 2
#define OPTIMIZATION 3
#define WEIGHTSWEEP_RATIOS 4
#define SINGLE_RATIOS 5
#define SINGLE_TIMESERIES 6 

#ifndef MODE 
#define MODE SINGLE_TIMESERIES 
#endif

#define TRUE 1
#define FALSE 0

#ifndef RETURN_COVS
#define RETURN_COVS FALSE 
#endif

using namespace std;

double tf;
int nreps;
const double dt = 0.01;
int n_groups;
double a,b,c,d; 
double wee, wei, wie, wii;
double sint, sext, total_noise, sig2noi;


mt19937 gen(54634134);
normal_distribution<double> ran_g(0.0, 1.0);

void simulate_system(const double tf, const double dt, vector<double> &corr_result);
void simulate_system(const double tf, const double dt, vector<vector<double>> &cov_matrix);
void simulate_system(const double tf, const double dt, ofstream &output);
void initial_conditions(vector<double> &E, vector<double> &I, double &sumE, double &sumI);

int main(int argc, char *argv[])
{
    int i,j;
    ofstream output;

    #if MODE==SINGLE
        //Process input from terminal
        if (argc==13)
        {
            tf       = stod(argv[1]);
            nreps    = stoi(argv[2]);
            n_groups = stoi(argv[3]);
            a        = stod(argv[4]);   
            b        = stod(argv[5]);  
            c        = stod(argv[6]);  
            d        = stod(argv[7]);  
            wee      = stod(argv[8]);  
            wei      = stod(argv[9]);  
            wie      = stod(argv[10]);  
            wii      = stod(argv[11]);  
            sig2noi  = stod(argv[12]);
        }
        else
        {
            cout << "INCORRECT NUMBER OF PARAMETERS" << endl;
            return EXIT_SUCCESS;
        }

        //Normalize constants to the value they should have...
        wee /= (n_groups-1);
        wei /= (n_groups-1);
        wie /= (n_groups-1);
        wii /= (n_groups-1);

        a -= wee;
        b -= wei;
        c -= wie;
        d -= wii;

        //Correlations do not depend on exact value of sint, but only on ratio. 
        //Then set a small value compared with linear coefficient to reduce need of statistics
        sint = a / 10.0; 
        sext = sig2noi * sint;

        #if RETURN_COVS==FALSE
            vector<double> corrs(4, 0.0);
            for (i=0; i < nreps; i++)
            {
                simulate_system(tf, dt, corrs);
            }

            cout << endl;
            for (i=0; i < corrs.size(); i++)
            {
                cout << corrs[i]/nreps << " ";
            }
        #else
            vector<vector<double>> corrs = vector<vector<double>>(2*n_groups, vector<double>(2*n_groups));
            simulate_system(tf, dt, corrs);


            output.open("../data/checks_tuning/cov_matrix");
            for (i=0; i < corrs.size(); i++)
            {
                for (j=0; j < corrs[i].size(); j++)
                {
                    output << corrs[i][j] << " ";
                }
                output << endl;
            }
            output.close();
        #endif
        cout << endl;

    #elif MODE==SINGLE_RATIOS
        double ree, rei, rie, rii;
        double coupling;

        //Process input from terminal
        if (argc==10)
        {
            tf          = stod(argv[1]);
            n_groups    = stoi(argv[2]);
            coupling    = stod(argv[3]);
            ree         = stod(argv[4]); 
            rei         = stod(argv[5]); 
            rie         = stod(argv[6]); 
            rii         = stod(argv[7]); 
            sig2noi     = stod(argv[8]);
            nreps       = stod(argv[9]);
        }
        else
        {
            cout << "INCORRECT NUMBER OF PARAMETERS" << endl;
            return EXIT_SUCCESS;
        }

        //Get the weights for everything
        a   = coupling*ree - 1; 
        b   = -coupling*rei;  
        c   = coupling*rie;
        d   = -coupling*rii - 1;
        wee = coupling*(1-ree);
        wei = -coupling*(1-rei);  
        wie = coupling*(1-rie);
        wii = -coupling*(1-rii); 

        //Then normalize as usual
        wee /= (n_groups-1);
        wei /= (n_groups-1);
        wie /= (n_groups-1);
        wii /= (n_groups-1);
        a -= wee;
        b -= wei;
        c -= wie;
        d -= wii;

        //Correlations do not depend on exact value of sint, but only on ratio. 
        //Then set a small value compared with linear coefficient 
        total_noise = a/10.0;
        sint = total_noise * (1-sig2noi);
        sext = total_noise * sig2noi; 

        #if RETURN_COVS==FALSE
            vector<double> corrs(4, 0.0);
            for (i=0; i < nreps; i++)
            {
                simulate_system(tf, dt, corrs);
            }

            cout << endl;
            for (i=0; i < corrs.size(); i++)
            {
                cout << corrs[i]/nreps << " ";
            }
        #else
            vector<vector<double>> corrs = vector<vector<double>>(2*n_groups, vector<double>(2*n_groups));
            simulate_system(tf, dt, corrs);


            output.open("../data/checks_tuning/cov_matrix");
            for (i=0; i < corrs.size(); i++)
            {
                for (j=0; j < corrs[i].size(); j++)
                {
                    output << corrs[i][j] << " ";
                }
                output << endl;
            }
            output.close();
        #endif
        cout << endl;
    #elif MODE==SINGLE_TIMESERIES
        double ree, rei, rie, rii;
        double coupling;

        //Process input from terminal
        if (argc==9)
        {
            tf          = stod(argv[1]);
            n_groups    = stoi(argv[2]);
            coupling    = stod(argv[3]);
            ree         = stod(argv[4]); 
            rei         = stod(argv[5]); 
            rie         = stod(argv[6]); 
            rii         = stod(argv[7]); 
            sig2noi     = stod(argv[8]);
        }
        else
        {
            cout << "INCORRECT NUMBER OF PARAMETERS" << endl;
            return EXIT_SUCCESS;
        }

        //Get the weights for everything
        a   = coupling*ree - 1; 
        b   = -coupling*rei;  
        c   = coupling*rie;
        d   = -coupling*rii - 1;
        wee = coupling*(1-ree);
        wei = -coupling*(1-rei);  
        wie = coupling*(1-rie);
        wii = -coupling*(1-rii); 


        //Then normalize as usual
        wee /= (n_groups-1);
        wei /= (n_groups-1);
        wie /= (n_groups-1);
        wii /= (n_groups-1);
        a -= wee;
        b -= wei;
        c -= wie;
        d -= wii;

        //Correlations do not depend on exact value of sint, but only on ratio. 
        //Then set a small value compared with linear coefficient 
        total_noise = a/10.0;
        sint = total_noise * (1-sig2noi);
        sext = total_noise * sig2noi; 

        simulate_system(tf, dt, output);
    #elif MODE==SIGNALSWEEP

        double dsratio;
        int npoints;

        //Process input from terminal
        if (argc==13)
        {
            tf        = stod(argv[1]);
            nreps     = stoi(argv[2]);
            n_groups  = stoi(argv[3]);
            a         = stod(argv[4]);   
            b         = stod(argv[5]);  
            c         = stod(argv[6]);  
            d         = stod(argv[7]);  
            wee       = stod(argv[8]);  
            wei       = stod(argv[9]);  
            wie       = stod(argv[10]);  
            wii       = stod(argv[11]);  
            npoints   = stoi(argv[12]);
        }
        else
        {
            cout << "INCORRECT NUMBER OF PARAMETERS" << endl;
            return EXIT_SUCCESS;
        }

        output.open("../data/correlations/corr_signalsweep-000INFO");
        output << "Self-interaction parameters: " << a << " " << b << " " << c << " " << d << endl;
        output << "Coupling parameters: " << wee << " " << wei << " " << wie << " " << wii << endl;
        output << "Time, number of repetitions, number of groups: " << tf << " " <<  nreps << " " << n_groups << endl; 
        output.close();

        //Correlations do not depend on exact value of sint, but only on ratio. 
        //Then set a small value compared with linear coefficient to reduce need of statistics
        total_noise = abs(a / 10.0);

        //Normalize constants to the value they should have...
        wee /= (n_groups-1);
        wei /= (n_groups-1);
        wie /= (n_groups-1);
        wii /= (n_groups-1);

        a -= wee;
        b -= wei;
        c -= wie;
        d -= wii;


        vector<double> corrs; //Initialize correlation vector
        dsratio = 1.0 / (1.0 * npoints); //Set the change of signal/noise ratio for each simulation

        //Prepare a file to spit all the results and compute them
        output.open("../data/correlations/corr_signalsweep_3");
        for (sig2noi = 0.0; sig2noi <= 1.0; sig2noi += dsratio)
        {
            sint = total_noise * (1-sig2noi);
            sext = total_noise * sig2noi; 

            cout << sig2noi << endl; 
            corrs = vector<double>(4, 0.0);
            for (i=0; i < nreps; i++) simulate_system(tf, dt, corrs);

            output << sig2noi << " ";
            for (i=0; i < corrs.size(); i++) output << corrs[i]/nreps << " ";
            output << endl;
        }
        output.close();

    #elif MODE==WEIGHTSWEEP

        double coupling;
        double coup_min, coup_max, dc;
        int npoints;

        //Process input from terminal
        if (argc==9)
        {
            tf          = stod(argv[1]);
            nreps       = stoi(argv[2]);
            n_groups    = stoi(argv[3]);
            coup_min    = stod(argv[4]);
            coup_max    = stod(argv[5]);
            npoints     = stoi(argv[6]);
            total_noise = stod(argv[7]);
            sig2noi     = stod(argv[8]);
        }
        else
        {
            cout << "INCORRECT NUMBER OF PARAMETERS" << endl;
            return EXIT_SUCCESS;
        }

        //Set noise intensity, which will be fixed during simulation
        sint = total_noise * (1-sig2noi);
        sext = total_noise * sig2noi; 

        //Write info file
        output.open("../data/correlations/corr_coupsweep-000INFO");
        output << "Coupling parameter from " << coup_min << " to " << coup_max << " in " << npoints << endl; 
        output << "Total noise, signal/noise ratio " << total_noise << " " << sig2noi << endl; 
        output << "Time, number of repetitions, number of groups: " << tf << " " <<  nreps << " " << n_groups << endl; 
        output.close();

        vector<double> corrs; //Initialize correlation vector
        dc = (coup_max - coup_min) / (1.0 * npoints); //Set the change of signal/noise ratio for each simulation

        //Prepare a file to spit all the results and compute them
        output.open("../data/correlations/corr_coupsweep2");
        for (coupling = coup_min; coupling <= coup_max; coupling += dc)
        {
            cout << coupling << endl; 

            //Set all weights depending on this coupling
            wee = wie = a = c =  coupling  / (n_groups-1);
            wei = wii = b = d = -coupling / (n_groups-1);
            a -= 1;
            d -= 1;

            a -= wee;
            b -= wei;
            c -= wie;
            d -= wii;


            //Do simulation
            corrs = vector<double>(4, 0.0);
            for (i=0; i < nreps; i++) simulate_system(tf, dt, corrs);

            output << coupling << " ";
            for (i=0; i < corrs.size(); i++) output << corrs[i]/nreps << " ";
            output << endl;
        }
        output.close();

    #elif MODE==WEIGHTSWEEP_RATIOS

        double coupling;
        double coup_min, coup_max, dc;
        double ree, rei, rie, rii;
        int npoints;
        string pathfile = "checks_tuning/coupsweep";

        //Process input from terminal
        if (argc==12)
        {
            tf          = stod(argv[1]);
            nreps       = stoi(argv[2]);
            n_groups    = stoi(argv[3]);
            coup_min    = stod(argv[4]);
            coup_max    = stod(argv[5]);
            npoints     = stoi(argv[6]);
            ree         = stod(argv[7]); 
            rei         = stod(argv[8]); 
            rie         = stod(argv[9]); 
            rii         = stod(argv[10]); 
            sig2noi     = stod(argv[11]);
        }
        else
        {
            cout << "INCORRECT NUMBER OF PARAMETERS" << endl;
            return EXIT_SUCCESS;
        }

        total_noise = 0.1;


        //Set noise intensity, which will be fixed during simulation
        sint = total_noise * (1-sig2noi);
        sext = total_noise * sig2noi; 

        //Write info file
        output.open("../data/"+pathfile+"-000INFO");
        output << "Coupling parameter from " << coup_min << " to " << coup_max << " in " << npoints << endl; 
        output << "Total noise, signal/noise ratio " << total_noise << " " << sig2noi << endl; 
        output << "Time, number of repetitions, number of groups: " << tf << " " <<  nreps << " " << n_groups << endl; 
        output.close();

        #if RETURN_COVS==FALSE
        vector<double> corrs; //Initialize correlation vector
        #else
        vector<vector<double>> corrs; //Initialize correlation vector
        #endif

        dc = (coup_max - coup_min) / (1.0 * npoints); //Set the change of signal/noise ratio for each simulation

        //Prepare a file to spit all the results and compute them
        output.open("../data/"+pathfile);
        for (coupling = coup_min; coupling <= coup_max; coupling += dc)
        {
            cout << coupling << endl; 

            //Use the rXY ratios to define the weights between internal and external inputs
            a   = coupling*ree - 1; 
            b   = -coupling*rei;  
            c   = coupling*rie;
            d   = -coupling*rii - 1;
            wee = coupling*(1-ree);
            wei = -coupling*(1-rei);  
            wie = coupling*(1-rie);
            wii = -coupling*(1-rii); 


            //Then normalize as usual
            wee /= (n_groups-1);
            wei /= (n_groups-1);
            wie /= (n_groups-1);
            wii /= (n_groups-1);
            a -= wee;
            b -= wei;
            c -= wie;
            d -= wii;

            //Do simulation
            corrs = vector<double>(4, 0.0);
            for (i=0; i < nreps; i++)
            {
                simulate_system(tf, dt, corrs);
            }

            cout << endl;
            for (i=0; i < corrs.size(); i++)
            {
                cout << corrs[i]/nreps << " ";
            }
            cout << endl;
        }
        output.close();
    #elif MODE==OPTIMIZATION
        //Process input from terminal
        if (argc==9)
        {
            tf       = stod(argv[1]);
            nreps    = stoi(argv[2]);
            n_groups = stoi(argv[3]);
            a        = stod(argv[4]);   
            b        = stod(argv[5]);  
            c        = stod(argv[6]);  
            d        = stod(argv[7]);  
            sig2noi  = stod(argv[8]);
        }
        else
        {
            cout << "INCORRECT NUMBER OF PARAMETERS" << endl;
            return EXIT_SUCCESS;
        }

        wee =  1.0 - a;
        wei = -1.0 - b;
        wie =  1.0 - c;
        wii = -1.0 - d;

        //Normalize constants to the value they should have...
        wee /= (n_groups-1);
        wei /= (n_groups-1);
        wie /= (n_groups-1);
        wii /= (n_groups-1);

        a -= 1.0 + wee;
        b -= wei;
        c -= wie;
        d -= 1.0 + wii;

        //Correlations do not depend on exact value of sint, but only on ratio. 
        //Then set a small value compared with linear coefficient to reduce need of statistics
        sint = a / 10.0; 
        sext = sig2noi * sint;

        vector<double> corrs(4, 0.0);
        for (i=0; i < nreps; i++)
        {
            simulate_system(tf, dt, corrs);
        }

        cout << endl;
        for (i=0; i < corrs.size(); i++)
        {
            cout << corrs[i]/nreps << " ";
        }
        cout << endl;
    #endif
    
    return EXIT_SUCCESS;
}





void initial_conditions(vector<double> &E, vector<double> &I, double &sumE, double &sumI)
{
    int i;

    uniform_real_distribution<double> ran_u(0.0, 1.0);

    //Get the initial conditions and initial sum of E and I
    sumE = sumI = 0.0;
    for (i=0; i < n_groups; i++)
    {
        E[i] = 0.2*ran_u(gen);
        I[i] = 0.1*ran_u(gen);
        sumE += E[i];
        sumI += I[i];
    }

    return;
}

void simulate_system(const double tf, const double dt, vector<double> &corr_result) 
{
    //Counters and declarations
    int i,j,k;
    double t, trelax;
    const int nits = int(tf/dt);

    trelax = tf / 5.0;

    //Stochastic dt
    double sqdt = sqrt(dt);

    //Number of connections in all-to-all connected (needed to know the number of i-j pairs for between group correlations)
    int n_links = n_groups*(n_groups-1)/2;

    //Observables and moments to be measured for each group
    vector<double> x, y, x2, y2, xy, x2_ext, y2_ext, xy_ext;
    x      = vector<double>(n_groups, 0.0);
    y      = vector<double>(n_groups, 0.0);
    x2     = vector<double>(n_groups, 0.0);
    y2     = vector<double>(n_groups, 0.0);
    xy     = vector<double>(n_groups, 0.0);
    x2_ext = vector<double>(n_links, 0.0);
    y2_ext = vector<double>(n_links, 0.0);
    xy_ext = vector<double>(n_links, 0.0);

    //Noise variables
    double noise_4_group;

    //Dynamical variables
    vector<double> E = vector<double>(n_groups);
    vector<double> I = vector<double>(n_groups);

    vector<double> oldE = vector<double>(n_groups);
    vector<double> oldI = vector<double>(n_groups);

    double sumE, sumI, old_sumE, old_sumI;

    //Initial conditions
    initial_conditions(E, I, sumE, sumI);
    
    //Integrate for relaxation
    for (t=0; t < trelax; t += dt)
    {
        //Swap variables with the old ones to avoid rewriting during update
        oldE.swap(E);
        oldI.swap(I);
        old_sumE = sumE;
        old_sumI = sumI;

        sumE = sumI = 0.0;
        for (j=0; j < n_groups; j++)
        {
            //External noise for this group
            noise_4_group = ran_g(gen);

            //Important: constants a,b,c,d have been renormalized to a - wee/(N-1) so the sum can be done for all N
            E[j] = oldE[j] + dt * (a*oldE[j] + b*oldI[j] + wee*old_sumE + wei*old_sumI) + sqdt * (sext * noise_4_group + sint * ran_g(gen));
            I[j] = oldI[j] + dt * (c*oldE[j] + d*oldI[j] + wie*old_sumE + wii*old_sumI) + sqdt * (sext * noise_4_group + sint * ran_g(gen));

            sumE += E[j];
            sumI += I[j];

        }
    } 


    //Integrate + observables
    for (t=0; t < tf; t += dt)
    {
        //Swap variables with the old ones to avoid rewriting during update
        oldE.swap(E);
        oldI.swap(I);
        old_sumE = sumE;
        old_sumI = sumI;


        sumE = sumI = 0.0;
        for (j=0; j < n_groups; j++)
        {
            //External noise for this group
            noise_4_group = ran_g(gen);

            //Important: constants a,b,c,d have been renormalized to a - wee/(N-1) so the sum can be done for all N
            E[j] = oldE[j] + dt * (a*oldE[j] + b*oldI[j] + wee*old_sumE + wei*old_sumI) + sqdt * (sext * noise_4_group + sint * ran_g(gen));
            I[j] = oldI[j] + dt * (c*oldE[j] + d*oldI[j] + wie*old_sumE + wii*old_sumI) + sqdt * (sext * noise_4_group + sint * ran_g(gen));

            sumE += E[j];
            sumI += I[j];

            //Compute all moments in-group
            x[j]  += E[j];
            x2[j] += E[j]*E[j];
            y[j]  += I[j];
            y2[j] += I[j]*I[j];
            xy[j] += E[j]*I[j];
        }

        //Between-group moments
        //This loop cannot be inside the other because we need all the x[i], x[j] before.
        k = 0; 
        for (j=0; j < n_groups; j++)
        {
            for (i=j+1; i < n_groups; i++)
            {
                x2_ext[k] += E[j]*E[i];
                y2_ext[k] += I[j]*I[i];
                xy_ext[k] += E[j]*I[i];   
                k++; 
            }
        }
    } 

    //Finish moment computation...
    vector<double> g2x, g2y, g2xy, g2xy_ext, g2x_ext, g2y_ext;
    g2x      = vector<double>(n_groups);
    g2y      = vector<double>(n_groups);
    g2xy     = vector<double>(n_groups);
    g2xy_ext = vector<double>(n_links);
    g2x_ext  = vector<double>(n_links);
    g2y_ext  = vector<double>(n_links);

    double cxy, cxy_ext, cx_ext, cy_ext;
    cxy = cx_ext = cy_ext = cxy_ext = 0.0;

    double avxy, avx, avy;

    avxy = avx = avy = 0.0;
    for (j = 0; j < n_groups; j++)
    {
        //1. Divide by number of iterations done
        x[j]  /= nits;
        y[j]  /= nits;
        x2[j] /= nits;
        y2[j] /= nits;
        xy[j] /= nits;

        //2. Get connected correlations
        g2x[j]   = x2[j] - x[j]*x[j];
        g2y[j]   = y2[j] - y[j]*y[j];
        g2xy[j]  = xy[j] - x[j]*y[j];

        //3. Correlation coefficient
        cxy += g2xy[j] / sqrt(g2x[j] * g2y[j]); 
    }

    //Repeat for the between-group thing
    //This loop cannot be integrated in the one before because we need to 
    //finish the average for x and y, since we will be using x[i] and x[j] to finish correlations
    k=0;
    for (j = 0; j < n_groups; j++)
    {
        for (i=j+1; i<n_groups; i++)
        {
            x2_ext[k] /= nits; 
            y2_ext[k] /= nits; 
            xy_ext[k] /= nits; 
            
            g2x_ext[k]  = x2_ext[k] - x[j]*x[i];
            g2y_ext[k]  = y2_ext[k] - y[j]*y[i];
            g2xy_ext[k] = xy_ext[k] - x[j]*y[i];

            cx_ext  += g2x_ext[k] / sqrt(g2x[j] * g2x[i]);
            cy_ext  += g2y_ext[k] / sqrt(g2y[j] * g2y[i]);
            cxy_ext += g2xy_ext[k] / sqrt(g2x[j] * g2y[i]); 
            k++;
        }
    }

    //Finish average over coefficients
    cxy     /= n_groups;
    cx_ext  /= n_links;
    cy_ext  /= n_links;
    cxy_ext /= n_links;



    //Plot results
    //cout << cxy << " " << cx_ext << " " << cy_ext << " " << cxy_ext << endl;
    corr_result[0] += cxy;
    corr_result[1] += cxy_ext;
    corr_result[2] += cx_ext;
    corr_result[3] += cy_ext;
}


void simulate_system(const double tf, const double dt, vector<vector<double>> &cov_matrix)
{
    //Counters and declarations
    int i,j,k;
    double t, trelax;
    const int nits = int(tf/dt);

    trelax = tf / 5.0;

    //Stochastic dt
    double sqdt = sqrt(dt);

    //Number of connections in all-to-all connected (needed to know the number of i-j pairs for between group correlations)
    //int n_links = n_groups*(n_groups-1);

    //Observables and moments to be measured for each group
    vector<double> x, y, x2, y2, xy; 
    vector<vector<double>> x2_ext, y2_ext, xy_ext;

    x      = vector<double>(n_groups, 0.0);
    y      = vector<double>(n_groups, 0.0);
    x2     = vector<double>(n_groups, 0.0);
    y2     = vector<double>(n_groups, 0.0);
    xy     = vector<double>(n_groups, 0.0);
    x2_ext = vector<vector<double>>(n_groups, vector<double>(n_groups, 0.0));
    y2_ext = vector<vector<double>>(n_groups, vector<double>(n_groups, 0.0));
    xy_ext = vector<vector<double>>(n_groups, vector<double>(n_groups, 0.0));

    //Noise variables
    double noise_4_group;

    //Dynamical variables
    vector<double> E = vector<double>(n_groups);
    vector<double> I = vector<double>(n_groups);

    vector<double> oldE = vector<double>(n_groups);
    vector<double> oldI = vector<double>(n_groups);

    double sumE, sumI, old_sumE, old_sumI;

    //Initial conditions
    initial_conditions(E, I, sumE, sumI);
    
    //Integrate for relaxation
    for (t=0; t < trelax; t += dt)
    {
        //Swap variables with the old ones to avoid rewriting during update
        oldE.swap(E);
        oldI.swap(I);
        old_sumE = sumE;
        old_sumI = sumI;

        sumE = sumI = 0.0;
        for (j=0; j < n_groups; j++)
        {
            //External noise for this group
            noise_4_group = ran_g(gen);

            //Important: constants a,b,c,d have been renormalized to a - wee/(N-1) so the sum can be done for all N
            E[j] = oldE[j] + dt * (a*oldE[j] + b*oldI[j] + wee*old_sumE + wei*old_sumI) + sqdt * (sext * noise_4_group + sint * ran_g(gen));
            I[j] = oldI[j] + dt * (c*oldE[j] + d*oldI[j] + wie*old_sumE + wii*old_sumI) + sqdt * (sext * noise_4_group + sint * ran_g(gen));

            sumE += E[j];
            sumI += I[j];

        }
    } 


    //Integrate + observables
    for (t=0; t < tf; t += dt)
    {
        //Swap variables with the old ones to avoid rewriting during update
        oldE.swap(E);
        oldI.swap(I);
        old_sumE = sumE;
        old_sumI = sumI;


        sumE = sumI = 0.0;
        for (j=0; j < n_groups; j++)
        {
            //External noise for this group
            noise_4_group = ran_g(gen);

            //Important: constants a,b,c,d have been renormalized to a - wee/(N-1) so the sum can be done for all N
            E[j] = oldE[j] + dt * (a*oldE[j] + b*oldI[j] + wee*old_sumE + wei*old_sumI) + sqdt * (sext * noise_4_group + sint * ran_g(gen));
            I[j] = oldI[j] + dt * (c*oldE[j] + d*oldI[j] + wie*old_sumE + wii*old_sumI) + sqdt * (sext * noise_4_group + sint * ran_g(gen));

            sumE += E[j];
            sumI += I[j];

            //Compute all moments in-group
            x[j]  += E[j];
            x2[j] += E[j]*E[j];
            y[j]  += I[j];
            y2[j] += I[j]*I[j];
            xy[j] += E[j]*I[j];
        }

        //Between-group moments
        //This loop cannot be inside the other because we need all the x[i], x[j] before.
        k = 0; 
        for (i=0; i < n_groups; i++)
        {
            for (j=0; j < n_groups; j++)
            {
                if (i==j) continue;

                x2_ext[i][j] += E[i]*E[j];
                y2_ext[i][j] += I[i]*I[j];
                xy_ext[i][j] += E[i]*I[j];   
                k++; 
            }
        }
    } 

    //Finish moment computation...
    vector<double> g2x, g2y, g2xy;
    vector<vector<double>> g2xy_ext, g2x_ext, g2y_ext;

    g2x      = vector<double>(n_groups);
    g2y      = vector<double>(n_groups);
    g2xy     = vector<double>(n_groups);
    g2xy_ext = vector<vector<double>>(n_groups, vector<double>(n_groups));
    g2x_ext  = vector<vector<double>>(n_groups, vector<double>(n_groups));
    g2y_ext  = vector<vector<double>>(n_groups, vector<double>(n_groups));

    double avxy, avx, avy;

    avxy = avx = avy = 0.0;
    for (j = 0; j < n_groups; j++)
    {
        //1. Divide by number of iterations done
        x[j]  /= nits;
        y[j]  /= nits;
        x2[j] /= nits;
        y2[j] /= nits;
        xy[j] /= nits;


        //2. Get connected correlations
        g2x[j]   = x2[j] - x[j]*x[j];
        g2y[j]   = y2[j] - y[j]*y[j];
        g2xy[j]  = (xy[j] - x[j]*y[j]);
    }

    //Repeat for the between-group thing
    //This loop cannot be integrated in the one before because we need to 
    //finish the average for x and y, since we will be using x[i] and x[j] to finish correlations
    k=0;
    for (i = 0; i < n_groups; i++)
    {
        cov_matrix[i][i] = 1.0; 
        cov_matrix[i+n_groups][i+n_groups] = 1.0; 
        cov_matrix[i][i+n_groups] = g2xy[i] / sqrt(g2x[i] * g2y[i]);
        cov_matrix[i+n_groups][i] = g2xy[i] / sqrt(g2x[i] * g2y[i]);

        for (j=0; j<n_groups; j++)
        {
            if (i==j) continue;

            x2_ext[i][j] /= nits; 
            y2_ext[i][j] /= nits; 
            xy_ext[i][j] /= nits; 

        }
    }

    for (i=0; i < n_groups; i++)
    {
        for (j=0; j<n_groups; j++)
        {
            if (i==j) continue;

            cov_matrix[i][j] = (x2_ext[i][j] - x[i]*x[j]) / sqrt(g2x[i] * g2x[j]);
            cov_matrix[i+n_groups][j+n_groups] = (y2_ext[i][j] - y[i]*y[j]) / sqrt(g2y[i] * g2y[j]);

            cov_matrix[i][j+n_groups] = (xy_ext[i][j] - x[i]*y[j]) / sqrt(g2x[i] * g2y[j]);
            cov_matrix[i+n_groups][j] = (xy_ext[j][i] - x[j]*y[i]) / sqrt(g2x[j] * g2y[i]);
        }
    }

}



void simulate_system(const double tf, const double dt, ofstream &output) 
{
    //Counters and declarations
    int i,j,k;
    double t, trelax;
    const int nits = int(tf/dt);

    trelax = tf / 5.0;

    //Stochastic dt
    double sqdt = sqrt(dt);

    //Number of connections in all-to-all connected (needed to know the number of i-j pairs for between group correlations)
    int n_links = n_groups*(n_groups-1)/2;

    //Noise variables
    double noise_4_group;

    //Dynamical variables
    vector<double> E = vector<double>(n_groups);
    vector<double> I = vector<double>(n_groups);

    vector<double> oldE = vector<double>(n_groups);
    vector<double> oldI = vector<double>(n_groups);

    double sumE, sumI, old_sumE, old_sumI;

    //Initial conditions
    initial_conditions(E, I, sumE, sumI);
    
    //Integrate for relaxation
    for (t=0; t < trelax; t += dt)
    {
        //Swap variables with the old ones to avoid rewriting during update
        oldE.swap(E);
        oldI.swap(I);
        old_sumE = sumE;
        old_sumI = sumI;

        sumE = sumI = 0.0;
        for (j=0; j < n_groups; j++)
        {
            //External noise for this group
            noise_4_group = ran_g(gen);

            //Important: constants a,b,c,d have been renormalized to a - wee/(N-1) so the sum can be done for all N
            E[j] = oldE[j] + dt * (a*oldE[j] + b*oldI[j] + wee*old_sumE + wei*old_sumI) + sqdt * (sext * noise_4_group + sint * ran_g(gen));
            I[j] = oldI[j] + dt * (c*oldE[j] + d*oldI[j] + wie*old_sumE + wii*old_sumI) + sqdt * (sext * noise_4_group + sint * ran_g(gen));

            sumE += E[j];
            sumI += I[j];

        }
    } 


    //Integrate + observables
    output.open("timeseries");
    for (t=0; t < tf; t += dt)
    {
        //Swap variables with the old ones to avoid rewriting during update
        oldE.swap(E);
        oldI.swap(I);
        old_sumE = sumE;
        old_sumI = sumI;


        sumE = sumI = 0.0;
        for (j=0; j < n_groups; j++)
        {
            //External noise for this group
            noise_4_group = ran_g(gen);

            //Important: constants a,b,c,d have been renormalized to a - wee/(N-1) so the sum can be done for all N
            E[j] = oldE[j] + dt * (a*oldE[j] + b*oldI[j] + wee*old_sumE + wei*old_sumI) + sqdt * (sext * noise_4_group + sint * ran_g(gen));
            I[j] = oldI[j] + dt * (c*oldE[j] + d*oldI[j] + wie*old_sumE + wii*old_sumI) + sqdt * (sext * noise_4_group + sint * ran_g(gen));

            sumE += E[j];
            sumI += I[j];
        }

        output << t << " " << sumE << " " << sumI << " " << E[0] << " " << I[0] << endl;
    } 
    output.close();
}
