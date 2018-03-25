//
//  main.cpp
//  Ass1
//
//  Created by Sen HUANG on 19/03/2018.
//  Copyright © 2018 Sen HUANG. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <math.h>
using namespace std;

const unsigned short MAXN = 500;         // Max neurons in any layer
const unsigned short MAXPATS = 5000;    // Max training patterns

// Data info
char filename[20];
unsigned short NumIPs;        // Number of input elements
unsigned short NumOPs;        // Number of output elements
unsigned short NumTrnPats;    // Number of patterns in training set
unsigned short NumTstPats;     // Number of patterns in test set


// MLP parameters
unsigned long NumIts;       // Maximum training iterations
unsigned short NumHN;       // Number of hidden layers
unsigned short NumNeurons;      // Number of neurons in hidden layer 1
short Ordering;    // Ordering type
float LrnRate;      // Learning rate
float Mtm1;         // Momentum (t-1)
float Mtm2;         // Momentum (t-2)
float ObjErr;       // Objective error

// MLP weights
float **WeightIH;       // Weights from input layer to the hidden layer
float **WeightHO;       // Weights from input layer to the output layer

#define rando() ((double)rand()/((double)RAND_MAX+1))

float **Aloc2DAry(unsigned short, unsigned short); // Allocates memory for 2D array
void Free2DAry(float**, unsigned short);    // Frees memory in 2D array

int main(int argc, const char * argv[])
{
    // read file
    char Line[500], Tmp[20];
    cout << "Enter data filename: ";
    cin >> filename;
    cin.ignore();
    ifstream fin;
    fin.open(filename);
    if(!fin.good())
    {
        cerr << "File not found!" << endl;
        exit(1);
    }
    do  //eat comments
    {
        fin.getline(Line, 500);
    }while(Line[0] == ';');
    sscanf(Line, "%s%hu", Tmp, &NumIPs);    // extract number of input elements from Line
    fin >> Tmp >> NumOPs;
    fin >> Tmp >> NumTrnPats;
    fin >> Tmp >> NumTstPats;
    fin >> Tmp >> NumIts;
    fin >> Tmp >> NumHN;
    fin >> Tmp >> NumNeurons;
    fin >> Tmp >> LrnRate;
    fin >> Tmp >> Mtm1;
    fin >> Tmp >> Mtm2;
    fin >> Tmp >> ObjErr;
    fin >> Tmp >> Ordering;
    cout << "Ordering: " << Ordering << endl;
    if(NumIPs < 1 || NumIPs > MAXN || NumOPs < 1 || NumOPs > MAXN ||
       NumTrnPats < 1 || NumTrnPats > MAXPATS || NumTrnPats < 1 ||
       NumTrnPats > MAXPATS || NumIts < 1 || NumIts > 20e6 ||
       NumNeurons < 0 || NumNeurons > 500 || LrnRate < 0 || LrnRate > 1 ||
       Mtm1 < 0 || Mtm1 > 10 || Mtm2 < 0 || Mtm2 > 10 ||
       ObjErr < 0 || ObjErr > 10 || Ordering < 0)
    {
        cout << "Invalid specs in data file!" << endl;
        exit(1);
    }
    // Loads data in
    float **IPTrnData = Aloc2DAry(NumTrnPats + 1, NumIPs + 1);
    float **OPTrnData = Aloc2DAry(NumTrnPats + 1, NumOPs + 1);
    float **IPTstData = Aloc2DAry(NumTstPats + 1, NumIPs + 1);
    float **OPTstData = Aloc2DAry(NumTstPats + 1, NumOPs + 1);
    unsigned short i, j;
    for(i = 1; i <= NumTrnPats; ++i)
    {
        for(j = 1; j <= NumIPs; ++j){
            fin >> IPTrnData[i][j];
            //IPTrnData[i][j] = IPTrnData[i][j]/100;
            //cout << "IPTrnData[" << i << "][" << j << "] = " << IPTrnData[i][j] << endl;
        }
        for(j = 1; j <= NumOPs; ++j){
            fin >> OPTrnData[i][j];
            //cout << "OPTrnData[" << i << "][" << j << "] = " << OPTrnData[i][j] << endl;
        }
    }
    for(i = 1; i <= NumTstPats; ++i)
    {
        for(j = 1; j <= NumIPs; ++j){
            fin >> IPTstData[i][j];
            //IPTstData[i][j] = IPTstData[i][j]/100;
            //cout << "IPTstData[" << i << "][" << j << "] = " << IPTstData[i][j] << endl;
        }
        for(j = 1; j <= NumOPs; ++j){
            fin >> OPTstData[i][j];
            //cout << "OPTstData[" << i << "][" << j << "] = " << OPTstData[i][j] << endl;
        }
    }
    fin.close();
    
    // TrainNet
    
    unsigned short *Ranpat = new unsigned short[NumTrnPats + 1];
    // Aloc2DAry
    WeightIH = Aloc2DAry(NumIPs + 1, NumNeurons + 1);   // W from input to hidden
    WeightHO = Aloc2DAry(NumNeurons + 1, NumOPs + 1);   // W from hidden to output
    
    //int i, j;       // for iteration
    unsigned short k, p, np, op;   // k => output, p => patterns, i => inputs, j => hiddens
    float Error = 0;
    float **SumH = Aloc2DAry(NumTrnPats + 1, NumNeurons + 1);     // Sums outputs for hidden layers
    float **Hidden = Aloc2DAry(NumTrnPats + 1, NumNeurons + 1);  // outputs of hidden layers after activates
    float **SumO = Aloc2DAry(NumTrnPats + 1, NumOPs + 1);    // Sums ouput for hidden layers
    float **Output = Aloc2DAry(NumTrnPats + 1, NumOPs + 1);;  // outputs of output layers after activates
    float *DeltaO = new float[NumOPs + 1];   // Delta of outputs
    float *SumDOW = new float[NumNeurons + 1];  // Sums delta of weights to neurons
    float *DeltaH = new float[NumNeurons + 1];  // Delta of neurons in hidden layers
    float **DeltaWeightIH = Aloc2DAry(NumIPs + 1, NumNeurons + 1);
    float **DeltaWeightHO = Aloc2DAry(NumNeurons + 1, NumOPs + 1);
    
    // Init wts between -0.5 and +0.5
    for(j = 1; j <= NumNeurons; ++j)    // initialize WeightIH and DeltaWeightIH
    {
        for(i = 0; i <= NumIPs; ++i)
        {
            DeltaWeightIH[i][j] = 0;
            WeightIH[i][j] = float(rand())/RAND_MAX - 0.5;
            //cout << "WeightIH[" << i << "][" << j << "] = " << WeightIH[i][j] << endl;
        }
    }
    for(k = 1; k <= NumOPs; ++k) {    // initialize WeightHO and DeltaWeightHO
        for(j = 0; j <= NumNeurons; ++j)
        {
            DeltaWeightHO[j][k] = 0;
            WeightHO[j][k] = float(rand())/RAND_MAX - 0.5;
            //cout << "WeightHO[" << j << "][" << k << "] = " << WeightHO[j][k] << endl;
        }
    }
    
    for(int epoch = 0 ; epoch <= NumIts ; ++epoch) {
        for(p = 1; p <= NumTrnPats; ++p)
        {    // randomize order of training patterns
            Ranpat[p] = p;
        }
        for(p = 1; p <= NumTrnPats; ++p)
        {
            np = p + rando() * (NumTrnPats + 1 - p);
            op = Ranpat[p];
            Ranpat[p] = Ranpat[np];
            Ranpat[np] = op ;
        }
        if(Ordering == 2)
        {
            np = rando() * NumTrnPats + 1;
            p = Ranpat[np];
            op = rando() * NumTrnPats + 1;
            Ranpat[np] = Ranpat[op];
            Ranpat[op] = p ;
        }
        Error = 0;
        for(np = 1; np <= NumTrnPats; ++np)     // p => cnt number of patterns
        {
            if(Ordering == 0)
                p = np;
            else if(Ordering == 1 || Ordering == 2)
                p = Ranpat[np];
            else if(Ordering > 2 && Ordering < 100)
                p = np;
            
            // Cal Output of hidden layers
            for(j = 1; j <= NumNeurons; ++j)     // j => cnt number of neurons in this layer
            {
                SumH[p][j] = WeightIH[0][j];
                for(i = 1; i <= NumIPs; ++i)     // i => cnt number of input elements
                    SumH[p][j] += IPTrnData[p][i] * WeightIH[i][j];
                Hidden[p][j] = 1 / (1 + exp(-SumH[p][j]));
            }
            
            // Compute output unit activations and errors
            for(k = 1; k <= NumOPs; ++k)     // i => cnt number of outputs
            {
                SumO[p][k] = WeightHO[0][k];
                for(j = 1; j <= NumNeurons; ++j)     // j => cnt number of output elements
                    SumO[p][k] += Hidden[p][j] * WeightHO[j][k];
                Output[p][k] = 1 / (1 + exp(-SumO[p][k]));
                // Sum Squared Error
                //Error += 0.5 * (OPTrnData[p][k] - Output[p][k]) * (OPTrnData[p][k] - Output[p][k]); // SSE
                Error -= (OPTrnData[p][k] * log(Output[p][k]) + (1.0 - OPTrnData[p][k]) * log(1.0 - Output[p][k]));   //Cross-Entropy Error
                
                //DeltaO[k] = (OPTrnData[p][k] - Output[p][k]) * Output[p][k] * (1 - Output[p][k]);   // Sigmoidal Outputs, SSE
                DeltaO[k] = OPTrnData[p][k] - Output[p][k];   // Sigmoidal Outputs, Cross-Entropy Error, Linear Outputs, SSE
                //cout << Output[k] << endl;
            }
            
            // Back-propagate errors to hidden layers
            for(j = 1; j <= NumNeurons; ++j)
            {
                SumDOW[j] = 0;
                for(k = 1; k <= NumOPs; ++k)
                    SumDOW[j] += WeightHO[j][k] * DeltaO[k];
                DeltaH[j] = SumDOW[j] * Hidden[p][j] * (1 - Hidden[p][j]);
            }
            
            // Update weights from input to hidden (WeightIH)
            for(j = 1; j <= NumNeurons; ++j)
            {
                DeltaWeightIH[0][j] = LrnRate * DeltaH[j] + Mtm1 * DeltaWeightIH[0][j];
                WeightIH[0][j] += DeltaWeightIH[0][j];
                for(i = 1; i <= NumIPs; ++i)
                {
                    DeltaWeightIH[i][j] = LrnRate * IPTrnData[p][i] * DeltaH[j] + Mtm1 * DeltaWeightIH[i][j];
                    WeightIH[i][j] += DeltaWeightIH[i][j];
                }
            }
            
            for(k = 1; k <= NumOPs; ++k)
            {
                DeltaWeightHO[0][k] = LrnRate * DeltaO[k] + Mtm1 * DeltaWeightHO[0][k];
                WeightHO[0][k] += DeltaWeightHO[0][k];
                for(j = 1; j <= NumNeurons; ++j)
                {
                    DeltaWeightHO[j][k] = LrnRate * Hidden[p][j] * DeltaO[k] + Mtm1 * DeltaWeightHO[j][k];
                    WeightHO[j][k] += DeltaWeightHO[j][k];
                }
            }
        }
        if( epoch%100 == 0 ) fprintf(stdout, "\nEpoch %-5d :   Error = %f", epoch, Error) ;
        if( Error < 0.0004 ) break ;  /* stop learning when 'near enough' */
    }

    
    fprintf(stdout, "\n\nNETWORK DATA\n") ;   // print network outputs
    for( i = 1 ; i <= NumIPs ; i++ ) {
        fprintf(stdout, "Input%-4d\t", i) ;
    }
    for( k = 1 ; k <= NumOPs ; k++ ) {
        fprintf(stdout, "Target%-4d\tOutput%-4d\t", k, k) ;
    }
    for( p = 1 ; p <= NumTrnPats ; p++ ) {
        fprintf(stdout, "\n%d\t", p) ;
        for( i = 1 ; i <= NumIPs ; i++ ) {
            fprintf(stdout, "%f\t", IPTrnData[p][i]) ;
        }
        for( k = 1 ; k <= NumOPs ; k++ ) {
            fprintf(stdout, "%f\t%f\t", OPTrnData[p][k], Output[p][k]) ;
        }
    }
    
    float cntRight = 0;
    
    // TestNet
    for(p = 1; p <= NumTstPats; ++p)
    {
        // Cal Output of hidden layers
        for(j = 1; j <= NumNeurons; ++j)     // j => cnt number of neurons in this layer
        {
            SumH[p][j] = WeightIH[0][j];
            for(i = 1; i <= NumIPs; ++i)     // i => cnt number of input elements
                SumH[p][j] += IPTstData[p][i] * WeightIH[i][j];
            Hidden[p][j] = 1 / (1 + exp(-SumH[p][j]));
        }
        
        for(k = 1; k <= NumOPs; ++k)     // i => cnt number of outputs
        {
            SumO[p][k] = WeightHO[0][k];
            for(j = 1; j <= NumNeurons; ++j)
            {  // j => cnt number of output elements
                SumO[p][k] += Hidden[p][j] * WeightHO[j][k];
            }
            Output[p][k] = 1 / (1 + exp(-SumO[p][k]));
            //cout << Output[k] << "  ";
            //cout << OPTstData[p][1] << endl;
        }
        if(Output[p][1] >= 0.5) Output[p][1] = 1;
        else Output[p][1] = 0;
        if(OPTstData[p][1] == Output[p][1])
            ++cntRight;
    }
    cout << "\nAccuracy: " << cntRight/NumTstPats * 100 << "%" << endl;

    
    Free2DAry(WeightIH, NumIPs + 1);
    Free2DAry(WeightHO, NumNeurons + 1);
    Free2DAry(IPTrnData, NumTrnPats + 1);
    Free2DAry(OPTrnData, NumTrnPats + 1);
    Free2DAry(IPTstData, NumTstPats + 1);
    Free2DAry(OPTstData, NumTstPats + 1);
    Free2DAry(SumH, NumTrnPats + 1);
    Free2DAry(SumO, NumTrnPats + 1);
    Free2DAry(Hidden, NumTrnPats + 1);
    Free2DAry(Output, NumTrnPats + 1);
    Free2DAry(DeltaWeightIH, NumIPs + 1);
    Free2DAry(DeltaWeightHO, NumNeurons + 1);
    delete [] DeltaH;
    DeltaH = NULL;
    delete [] DeltaO;
    DeltaO = NULL;
    delete [] SumDOW;
    SumDOW = NULL;
    delete [] Ranpat;
    Ranpat = NULL;


    return 0;
}


// Allocates memory for 2D array
float **Aloc2DAry(unsigned short m, unsigned short n)
{
    float **Ary2D = new float*[m];
    if(NULL == Ary2D)
    {
        cerr << "No memory!" << endl;
        exit(1);
    }
    for(unsigned short i = 0; i < m; ++i)
    {
        Ary2D[i] = new float[n];
        if(NULL == Ary2D)
        {
            cerr << "No memory!" << endl;
            exit(1);
        }
    }
    return Ary2D;
}

// Frees memory in 2D array
void Free2DAry(float **Ary2D, unsigned short n)
{
    for(unsigned short i = 0; i < n; ++i)
    {
        delete [] Ary2D[i];
        Ary2D[i] = NULL;
    }
    delete [] Ary2D;
    Ary2D = NULL;
}

