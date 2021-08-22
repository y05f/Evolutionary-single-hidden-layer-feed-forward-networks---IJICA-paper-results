//2013-12-22 01:33:39
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <time.h>

using namespace std;
//-----------------------Structure Individu-----------------------
struct individual
{
  vector<vector<vector<float>>> netParam;
  vector<vector<vector<float>>> SP;
  vector<vector<float>> netBiais;
  vector<vector<float>> SPbiais;
  vector<vector<float>> neurOut;
  float fit;
  float MSEtrain;
  float MSEvalid;
  float MSEtest;
  float WTAtrain;
  float WTAvalid;
  float WTAtest;
};
//----------------------Les tableaux, variables et constantes utilisées------------------------
vector<individual> mu;            //Population des parents
vector<individual> lambda;        //Population des enfants
const int popsize = 20;           //Taille de la population courante
const int maxN = 50;              //Nombre maximal des neurones
const int minN = 1;               //Nombre minimal des neurones
const int prodcutionRatio = 4;    // lambda=prodcutionRatio*mu
const int totalGenerations = 500; //Nombre maximal de generations
int stop;                         //Nombre maximal de generations
const float limit = 0.5;          // les valeurs d'initialisation des poids synaptiques [-limit , +limit]
const float probNM = 0.4;         //Probabilité de mutation des neurones
const float probNR = 0.7;         //Probabilité de mutation des réseaux
const float probBP = 0.6;
const float etaBP = 0.25;
const float eps = 0.01;
const float beta = 0.4;
const int iterBP = 1;
const int shuffling = 0;
char selectionType = 'n';
string dataname;
vector<vector<float>> trainDo;
vector<vector<float>> trainIn;
vector<vector<float>> validDo;
vector<vector<float>> validIn;
vector<vector<float>> testDo;
vector<vector<float>> testIn;
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-----------Les fonctions utilisées pour la construction du réseau----------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
float f(float t, float v, const float beta) { return (beta * v + (1 - beta) * t); } // fonction fitness
//End of function
float sig(float f) { return (1. / (1. + exp(-f))); } // fonction sigmoide
//End of function
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
/*float tanh(float f) { return ((1-exp(-2*f))/(1+exp(-2*f)));}// fonction Hyperbolic tangent
//End of function*/
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
float randlimit(float limite)
{
  return (2. * (rand() % (1100000000) / 1000000000.) - 1.) * limite; // fonction retourne un nombre aleatoire entre -limite et + limite
} //End of function
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
float randMaxMin(float max, float min)
{
  return ((((float)rand()) / ((float)RAND_MAX)) * (max - min) + min);
} //End of function
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void netcalcul(vector<float> Inputs, individual &P)
{                                      // fonction qui calcule les sorties du réseau
  P.neurOut.resize(P.netParam.size()); // Actualisation de nombre de couches de neurones existantes dans le réseau
  for (int i = 0; i < P.neurOut.size(); i++)
    P.neurOut[i].resize(P.netParam[i].size(), 0); // Actualisation et initialisation des sorties de chaque neurone du réseau
  vector<vector<float>> fctsom(P.neurOut);        // fonction somme

  for (int i = 0; i < P.netParam.size(); i++)
    for (int j = 0; j < P.netParam[i].size(); j++)
    {
      fctsom[i][j] = P.netBiais[i][j];
      for (int l = 0; l < P.netParam[i][j].size(); l++)
      {
        if (i == 0 && Inputs.size() == P.netParam[i][j].size())
          fctsom[i][j] += Inputs[l] * P.netParam[i][j][l];
        if (i > 0)
          fctsom[i][j] += P.neurOut[i - 1][j] * P.netParam[i][j][l];
      }
      P.neurOut[i][j] = sig(fctsom[i][j]);
    }
  fctsom.clear();
} //End of function
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
float WTA(individual &P, vector<vector<float>> AllIn, vector<vector<float>> AllOutD)
{ // Winner takes all utilisé pour le calcul du pourcentage d'erreur de classification
  float WTAerr = 0, tmp = 0;
  vector<vector<float>> WTA(AllOutD);
  for (int i = 0; i < AllOutD.size(); i++)
    for (int j = 0; j < AllOutD[i].size(); j++)
      WTA[i][j] = -1; // Initialisation du vecteur des erreurs
  for (int i = 0; i < AllIn.size(); i++)
  {
    netcalcul(AllIn[i], P);
    for (int j = 0; j < AllOutD[i].size(); j++)
      if (P.neurOut[P.neurOut.size() - 1][tmp] < P.neurOut[P.neurOut.size() - 1][j])
        tmp = j; // sauvegarde de l'indice du neurone et non pas le contenu
    for (int l = 0; l < WTA[i].size(); l++)
    {
      if (l == tmp)
        WTA[i][l] = 1; // comparaison de l'indice du neurone de sortie avec l'indice de la sortie désirée
      else
        WTA[i][l] = 0;
    }
  }
  float ErrCount = 0;
  for (int i = 0; i < AllOutD.size(); i++)
    for (int j = 0; j < AllOutD[i].size(); j++)
      if ((WTA[i][j] - AllOutD[i][j]) != 0)
      {
        ErrCount++;
        break;
      }
  WTAerr = 100. * ErrCount / AllIn.size();
  WTA.clear();
  return WTAerr;
} //End of function
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
float MSE(individual &P, vector<vector<float>> AllIn, vector<vector<float>> AllOutD)
{
  float E = 0, tmp = 0;
  vector<float> e(AllIn.size(), 0);
  for (int i = 0; i < AllIn.size(); i++)
  {
    netcalcul(AllIn[i], P);
    for (int j = 0; j < AllOutD[i].size(); j++)
    {
      e[i] += pow(AllOutD[i][j] - P.neurOut[P.neurOut.size() - 1][j], 2);
    }
  }
  for (int k = 0; k < e.size(); k++)
    tmp += e[k];
  E = 100. * tmp / ((float)AllIn.size() * (float)AllOutD[0].size()); // equation de Prechlet "Proben1"
  e.clear();
  return E;
} //End of function
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
/*void shuffleData(vector<vector<float> > &AllIn,vector<vector<float> > &AllOutD){
  int tmp;
  if(AllIn.size()!=AllOutD.size()) {cout<<"shuffle function err"<<endl; exit(1);}
  for(int i=0;i<AllIn.size();i++){
    do{tmp=rand()%AllIn.size();}while(tmp==i);
    swap(AllIn[i],AllIn[tmp]);
    swap(AllOutD[i],AllOutD[tmp]);
  }
}*/
//End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void backpropagation(vector<vector<float>> AllIn, vector<vector<float>> AllOutD, individual &P, const float eta, const int StopIter, const int s)
{
  int t = 0;
  P.neurOut.resize(P.netParam.size());
  for (int i = 0; i < P.neurOut.size(); i++)
    P.neurOut[i].resize(P.netParam[i].size(), 0); // Initialisation du vecteur des sorties des neurones
  vector<vector<float>> delta(P.neurOut);
  do
  {
    //if(s==1) shuffleData(AllIn,AllOutD);
    for (int i = 0; i < AllIn.size(); i++)
    {
      //1)Propagation des signaux:
      netcalcul(AllIn[i], P);
      //2)Calcul du Delta de chaque neurone:
      for (int j = 0; j < delta[delta.size() - 1].size(); j++) // Pour calculer Delta des neurones de la couche de sortie
        delta[delta.size() - 1][j] = P.neurOut[P.neurOut.size() - 1][j] * (1 - P.neurOut[P.neurOut.size() - 1][j]) * (AllOutD[i][j] - P.neurOut[P.neurOut.size() - 1][j]);
      for (int j = P.netParam.size() - 2; j > -1; j--)
      { // Pour calculer Delta des neurones des couches cachées
        for (int k = 0; k < P.netParam[j].size(); k++)
        {
          float sommeDelta = 0;
          for (int l = 0; l < P.netParam[j + 1].size(); l++)
            for (int m = 0; m < P.netParam[j + 1][l].size(); m++)
              sommeDelta += delta[j + 1][l] * P.netParam[j + 1][l][m];
          delta[j][k] = P.neurOut[j][k] * (1 - P.neurOut[j][k]) * sommeDelta;
        }
      }
      //3)Correction des poids synaptiques:
      for (int j = 0; j < P.netParam.size(); j++)
        for (int k = 0; k < P.netParam[j].size(); k++)
        {
          P.netBiais[j][k] += eta * delta[j][k];
          for (int l = 0; l < P.netParam[j][k].size(); l++)
          {
            if (j == 0)
              P.netParam[j][k][l] += eta * delta[j][k] * AllIn[i][l];
            else
              P.netParam[j][k][l] += eta * delta[j][k] * P.neurOut[j - 1][l];
          }
        }
    } //Fin de la boucle(i)
    t++;
  } while (t < StopIter);
} //End of function
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void PopBP(vector<vector<float>> AllIn, vector<vector<float>> AllOutD, vector<individual> &P, const float Prob, const float eta, const int StopIter)
{
  int i = 0, k = P.size();
#pragma omp parallel for schedule(dynamic)
  for (i = 0; i < k; i++)
  {
    if (randMaxMin(0, 1) <= Prob && P[i].netParam[0].size() != 0)
      backpropagation(AllIn, AllOutD, P[i], eta, StopIter, shuffling);
  } /*-- End of parallel region --*/
} //End of function
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
float MSEAtest(vector<individual> &P)
{
  float result = 0;

  for (int k = 0; k < P.size(); k++)
    result += P[k].MSEtest;
  return result /= P.size();
} //End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
float MSEAvalid(vector<individual> &P)
{
  float result = 0;

  for (int k = 0; k < P.size(); k++)
    result += P[k].MSEvalid;
  return result /= P.size();
} //End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
float MSEAtrain(vector<individual> &P)
{
  float result = 0;

  for (int k = 0; k < P.size(); k++)
    result += P[k].MSEtrain;
  return result /= P.size();
} //End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-----------Les fonctions de l'algorithme d'evolution utilisé-------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void initialize(vector<individual> &P)
{
  int i = 0, k = P.size();
#pragma omp parallel for schedule(dynamic) // Allocation de la couche cachée et de sortie
  for (i = 0; i < k; i++)
    P[i].netParam.resize(2);
#pragma omp parallel for schedule(dynamic) // Allocation des neurones des deux couches
  for (i = 0; i < k; i++)
  {
    P[i].netParam[0].resize(rand() % (maxN - minN) + minN);
    P[i].netParam[1].resize(trainDo[0].size());
  }
/*-- End of parallel region --*/
#pragma omp parallel for schedule(dynamic) // Allocation des poids synaptiques
  for (i = 0; i < k; i++)
  {
    for (int j = 0; j < P[i].netParam.size(); j++)
      for (int l = 0; l < P[i].netParam[j].size(); l++)
        if (j == 0)
          P[i].netParam[j][l].resize(trainIn[0].size());
        else
          P[i].netParam[j][l].resize(P[i].netParam[j - 1].size());
  }                                        /*-- End of parallel region --*/
#pragma omp parallel for schedule(dynamic) // Allocation des biais des neurones et des paramètres de stratégies de l'AE
  for (i = 0; i < k; i++)
  {
    P[i].SP.resize(P[i].netParam.size());
    P[i].SPbiais.resize(P[i].netParam.size());
    P[i].netBiais.resize(P[i].netParam.size());
    for (int j = 0; j < P[i].netParam.size(); j++)
    {
      P[i].SP[j].resize(P[i].netParam[j].size());
      P[i].SPbiais[j].resize(P[i].netParam[j].size());
      P[i].netBiais[j].resize(P[i].netParam[j].size());
      for (int l = 0; l < P[i].netParam[j].size(); l++)
        P[i].SP[j][l].resize(P[i].netParam[j][l].size());
    }
    // Initialisation
    P[i].fit = 0;
    P[i].MSEtrain = 0;
    P[i].MSEtest = 0;

  } /*-- End of parallel region --*/
  default_random_engine generator;
  normal_distribution<float> distribution(0.0, 1.0);
  default_random_engine generator1;
  normal_distribution<float> distribution1(0.0, 1.0);
#pragma omp parallel for schedule(dynamic) // Initialisation des poids et des biais de la population
  for (i = 0; i < k; i++)
    for (int j = 0; j < P[i].netParam.size(); j++)
      for (int l = 0; l < P[i].netParam[j].size(); l++)
      {
        P[i].SPbiais[j][l] = distribution1(generator1);
        P[i].netBiais[j][l] = randlimit(limit);
        for (int m = 0; m < P[i].netParam[j][l].size(); m++)
        {
          P[i].netParam[j][l][m] = randlimit(limit);
          P[i].SP[j][l][m] = distribution(generator);
        }
      } /*-- End of parallel region --*/
} //End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void fitness(vector<individual> &P)
{
  int n = P.size();
  int k = 0;
#pragma omp parallel for schedule(dynamic)
  for (k = 0; k < n; k++)
  {
    P[k].MSEtrain = MSE(P[k], trainIn, trainDo);
    P[k].MSEvalid = MSE(P[k], validIn, validDo);
    P[k].MSEtest = MSE(P[k], testIn, testDo);

    P[k].WTAtrain = WTA(P[k], trainIn, trainDo);
    P[k].WTAvalid = WTA(P[k], validIn, validDo);
    P[k].WTAtest = WTA(P[k], testIn, testDo);

    P[k].fit = f(P[k].MSEtrain, P[k].MSEvalid, beta);
  } /*-- End of parallel region --*/
} //End of function
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void sort(vector<individual> &P, char o)
{
  for (int j = 0; j < P.size(); j++)
    for (int i = 0; i < P.size() - 1; i++)
    {
      if (o == 'c' && P[i].fit > P[i + 1].fit)
        iter_swap(P.begin() + i, P.begin() + i + 1); //library <algorithm>
      if (o == 'd' && P[i].fit < P[i + 1].fit)
        iter_swap(P.begin() + i, P.begin() + i + 1); //library <algorithm>
    }
} //End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void Create_Mu_Lambda(vector<individual> &P, vector<individual> &P1, vector<individual> &P2, int size)
{
  P1 = P;
  P2 = P1;
  for (int i = 0; i < size - 1; i++)
    P2.insert(P2.end(), P1.begin(), P1.end());
} //End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void spMutation(vector<individual> &P)
{
  int k = P.size();
  float tau0 = (1 / sqrt(2 * popsize));
  float tau = (1 / sqrt(2 * sqrt(popsize)));
  float xi0, xi, xi0b, xib;
  default_random_engine generator;
  normal_distribution<float> distribution(0.0, 1.0);
  default_random_engine generator1;
  normal_distribution<float> distribution1(0.0, 1.0);
  int i = 0;
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < k; i++)
  {
    xi0 = tau0 * distribution(generator);
    xi0b = tau0 * distribution1(generator1);
    for (int j = 0; j < P[i].SP.size(); j++)
      for (int l = 0; l < P[i].SP[j].size(); l++)
      {
        xib = tau * distribution1(generator1);
        P[i].SPbiais[j][l] = P[i].SPbiais[j][l] * exp(xib) * exp(xi0b);
        for (int m = 0; m < P[i].SP[j][l].size(); m++)
        {
          xi = tau * distribution(generator);
          P[i].SP[j][l][m] = P[i].SP[j][l][m] * exp(xi) * exp(xi0);
        }
      }
  } /*-- End of parallel region --*/
} //End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void opMutation(vector<individual> &P)
{
  int k = P.size();
  default_random_engine generator;
  normal_distribution<float> distribution(0.0, 1.0);
  default_random_engine generator1;
  normal_distribution<float> distribution1(0.0, 1.0);
  int i = 0;
#pragma omp parallel for schedule(dynamic)
  for (i = 0; i < k; i++)
    for (int j = 0; j < P[i].netParam.size(); j++)
      for (int l = 0; l < P[i].netParam[j].size(); l++)
      {
        P[i].netBiais[j][l] = P[i].netBiais[j][l] + P[i].SPbiais[j][l] * distribution1(generator1);
        for (int m = 0; m < P[i].netParam[j][l].size(); m++)
          P[i].netParam[j][l][m] = P[i].netParam[j][l][m] + P[i].SP[j][l][m] * distribution(generator);
      }
  /*-- End of parallel region --*/
} //End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void NeurMutation(vector<individual> &P)
{
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < P.size(); i++)
  {
    if (randMaxMin(0, 1) <= probNM && P[i].netParam[0].size() != 0)
    {
      int x;
      x = rand() % P[i].netParam[0].size(); //L'emplacement du neurone à ajouter
      if (randMaxMin(0, 1) < 0.5)
      {
        vector<float> tmp;
        vector<float> stmp;
        default_random_engine generator;
        normal_distribution<float> distribution(0.0, 1.0);
        for (int l = 0; l < trainIn[0].size(); l++)
        { //générer les poids du nouveau neurone et leurs SP
          tmp.push_back(randlimit(limit));
          stmp.push_back(distribution(generator));
        }
        //Ajout du neurone
        P[i].netParam[0].insert(P[i].netParam[0].begin() + x, tmp);
        P[i].SP[0].insert(P[i].SP[0].begin() + x, stmp);
        P[i].netBiais[0].insert(P[i].netBiais[0].begin() + x, randlimit(limit));
        P[i].SPbiais[0].insert(P[i].SPbiais[0].begin() + x, distribution(generator));
        for (int j = 0; j < P[i].netParam[1].size(); j++)
        {
          P[i].netParam[1][j].insert(P[i].netParam[1][j].begin() + x, randlimit(limit));
          P[i].SP[1][j].insert(P[i].SP[1][j].begin() + x, distribution(generator));
        }
        tmp.clear();
        stmp.clear();
      }
      else
      {
        //suppression du neurone
        P[i].netParam[0].erase(P[i].netParam[0].begin() + x);
        P[i].SP[0].erase(P[i].SP[0].begin() + x);
        P[i].netBiais[0].erase(P[i].netBiais[0].begin() + x);
        P[i].SPbiais[0].erase(P[i].SPbiais[0].begin() + x);
        for (int j = 0; j < P[i].netParam[1].size(); j++)
        {
          P[i].netParam[1][j].erase(P[i].netParam[1][j].begin() + x);
          P[i].SP[1][j].erase(P[i].SP[1][j].begin() + x);
        }
      }
    }
  }
} //End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void NeurRecombination(vector<individual> &P)
{ // opérateur de croisement de deux couches cachées
  int b, couple = 0;

  for (int i = 0; i < P.size(); i++)
  {
    if (randMaxMin(0, 1) <= probNR && P[i].netParam[0].size() != 0)
    {
      couple++;
      if (couple % 2 != 0)
        b = i;
      else
      {
        individual Ptmpi = P[i], Ptmpb = P[b];
        int pt1, pt2;
        pt1 = rand() % P[i].netParam[0].size();
        pt2 = rand() % P[b].netParam[0].size();
        //Suppression des 2 parties pt1 et pt2
        Ptmpi.netParam[0].erase(Ptmpi.netParam[0].begin(), Ptmpi.netParam[0].begin() + pt1);
        Ptmpi.SP[0].erase(Ptmpi.SP[0].begin(), Ptmpi.SP[0].begin() + pt1);
        Ptmpi.netBiais[0].erase(Ptmpi.netBiais[0].begin(), Ptmpi.netBiais[0].begin() + pt1);
        Ptmpi.SPbiais[0].erase(Ptmpi.SPbiais[0].begin(), Ptmpi.SPbiais[0].begin() + pt1);
        for (int j = 0; j < Ptmpi.netParam[1].size(); j++)
        {
          Ptmpi.netParam[1][j].erase(Ptmpi.netParam[1][j].begin(), Ptmpi.netParam[1][j].begin() + pt1);
          Ptmpi.SP[1][j].erase(Ptmpi.SP[1][j].begin(), Ptmpi.SP[1][j].begin() + pt1);
        }
        Ptmpb.netParam[0].erase(Ptmpb.netParam[0].begin(), Ptmpb.netParam[0].begin() + pt2);
        Ptmpb.SP[0].erase(Ptmpb.SP[0].begin(), Ptmpb.SP[0].begin() + pt2);
        Ptmpb.netBiais[0].erase(Ptmpb.netBiais[0].begin(), Ptmpb.netBiais[0].begin() + pt2);
        Ptmpb.SPbiais[0].erase(Ptmpb.SPbiais[0].begin(), Ptmpb.SPbiais[0].begin() + pt2);
        for (int j = 0; j < Ptmpb.netParam[1].size(); j++)
        {
          Ptmpb.netParam[1][j].erase(Ptmpb.netParam[1][j].begin(), Ptmpb.netParam[1][j].begin() + pt2);
          Ptmpb.SP[1][j].erase(Ptmpb.SP[1][j].begin(), Ptmpb.SP[1][j].begin() + pt2);
        }
        //Ajout des 2 parties pt1 et pt2
        Ptmpi.netParam[0].insert(Ptmpi.netParam[0].begin(), P[b].netParam[0].begin(), P[b].netParam[0].begin() + pt2);
        Ptmpi.SP[0].insert(Ptmpi.SP[0].begin(), P[b].SP[0].begin(), P[b].SP[0].begin() + pt2);
        Ptmpi.netBiais[0].insert(Ptmpi.netBiais[0].begin(), P[b].netBiais[0].begin(), P[b].netBiais[0].begin() + pt2);
        Ptmpi.SPbiais[0].insert(Ptmpi.SPbiais[0].begin(), P[b].SPbiais[0].begin(), P[b].SPbiais[0].begin() + pt2);
        for (int j = 0; j < Ptmpi.netParam[1].size(); j++)
        {
          Ptmpi.netParam[1][j].insert(Ptmpi.netParam[1][j].begin(), P[b].netParam[1][j].begin(), P[b].netParam[1][j].begin() + pt2);
          Ptmpi.SP[1][j].insert(Ptmpi.SP[1][j].begin(), P[b].SP[1][j].begin(), P[b].SP[1][j].begin() + pt2);
        }

        Ptmpb.netParam[0].insert(Ptmpb.netParam[0].begin(), P[i].netParam[0].begin(), P[i].netParam[0].begin() + pt1);
        Ptmpb.SP[0].insert(Ptmpb.SP[0].begin(), P[i].SP[0].begin(), P[i].SP[0].begin() + pt1);
        Ptmpb.netBiais[0].insert(Ptmpb.netBiais[0].begin(), P[i].netBiais[0].begin(), P[i].netBiais[0].begin() + pt1);
        Ptmpb.SPbiais[0].insert(Ptmpb.SPbiais[0].begin(), P[i].SPbiais[0].begin(), P[i].SPbiais[0].begin() + pt1);
        for (int j = 0; j < Ptmpb.netParam[1].size(); j++)
        {
          Ptmpb.netParam[1][j].insert(Ptmpb.netParam[1][j].begin(), P[i].netParam[1][j].begin(), P[i].netParam[1][j].begin() + pt1);
          Ptmpb.SP[1][j].insert(Ptmpb.SP[1][j].begin(), P[i].SP[1][j].begin(), P[i].SP[1][j].begin() + pt1);
        }
        P[i] = Ptmpi;
        P[b] = Ptmpb;
      }
    }
  }
} //End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void showpop(vector<individual> &P)
{
  sort(P, 'c');
  system("clear");
  cout << stop << endl;
  for (int i = 0; i < P.size(); i++)
  {
    cout.precision(4);
    cout << "individu N°" << i << "\tNeurNumb=" << P[i].netParam[0].size() << "\tfit=" << P[i].fit << "\ttrainMSE=" << P[i].MSEtrain << "\tvalidMSE=" << P[i].MSEvalid << "\ttestMSE=" << P[i].MSEtest << endl;
  }
} //End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void selection(vector<individual> &P, vector<individual> &P1, vector<individual> &P2, float Spressure, char t)
{ // fonction de sélection option p= par rang avec pression, r= par rang standard, n= sévere(normale les n meilleurs)
  vector<individual> tmppop;
  tmppop = P1;
  tmppop.insert(tmppop.end(), P2.begin(), P2.end());
  //showpop(P1);
  /*  if(t=='p') {
    //showpop(tmppop);
    sort(tmppop,'d');
    int l=0;
    while(l<P.size()){
      for(int i=tmppop.size()-1;i>-1;i--)
        if(randMaxMin(0,2)<=(2-Spressure+2*(Spressure-1)*(float)i/(tmppop.size()-1)) && l<P.size() )  {P[l]=tmppop[i];l++;}
    }
  }
  else if(t=='r') {
    //showpop(tmppop);
    sort(tmppop,'d');
    vector<float> fitPos(tmppop.size());
    int l=0;
    while(l<P.size()){
      for(int i=tmppop.size()-1;i>-1;i--)
        if(randMaxMin(0,1)<=((float)i/tmppop.size()) && l<P.size() )  {P[l]=tmppop[i];l++;}
    }
  }
  else*/
  if (t == 'n')
  {
    sort(tmppop, 'c');
    //showpop(tmppop);
    for (int i = 0; i < P.size(); i++)
      P[i] = tmppop[i];
  }
  else
  {
    cout << "erreur de selection" << endl;
    exit(1);
  }
  tmppop.clear();
} //End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
/*void savepop(vector<individual> &P){// fonction de sauvegarde de la population entière à chaque génération
  //sort(P,'c');
  string dos="./save/";
  /*string path="mkdir ";
  dos.append(dataname);
  path.append(dos);
 system(path.c_str());
  for(int i=0;i<P.size();i++){
    string link;
    link=dos;
    ostringstream oss1,oss;
    oss1<<i;
    oss << P[i].netParam[0].size();
    string result = oss.str();
    link.append("/");
    link.append(oss1.str());
    link.append("-");
    link.append(result);
    link.append("-SavedWeights.xls");
    ofstream file(link.c_str());
    file<<"individu:"<<i<<",fit="<<P[i].fit<<"\ttrainMSE="<<P[i].MSEtrain<<"\ttestMSE="<<P[i].MSEtest<<"*\n";
    for(int j=0;j<P[i].netParam.size();j++)
      for(int l=0;l<P[i].netParam[j].size();l++){
        file<<"C"<<j<<",";
        file<<"N"<<l<<",";
        file<<P[i].netBiais[j][l]<<",";
        for(int m=0;m<P[i].netParam[j][l].size();m++)
          file<<P[i].netParam[j][l][m]<<",";
          file<<"\n";
      }
    file.close();
    link.clear();
    result.clear();
  }*/
/*string link2(dos);
  link2.append("/results.xls");
  ofstream file1(link2.c_str(), ios::app);
  file1<<"\n+++++++G+e+n+e+r++"<<stop<<"+a+t+i+o+n+++++++++++"<<endl;
  file1<<"individu Num;Fit;trainMSE;validMSE;testMSE"<<endl;
  for(int i=0;i<P.size();i++){
    file1<<i<<";"<<P[i].fit<<";"<<P[i].MSEtrain<<";"<<P[i].MSEvalid<<";"<<P[i].MSEtest<<endl;}
  file1<<";"<<"Mean;"<<MSEAtrain(P)<<";"<<MSEAvalid(P)<<";"<<MSEAtest(P)<<endl;

  file1.close();

  dos.clear();
  //path.clear();


}*/
//End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
bool FileExists(const char *FileName)
{ // fonction qui vérifie l'existance d'un fichier
  FILE *fp = NULL;
  fp = fopen(FileName, "rb");
  if (fp != NULL)
  {
    fclose(fp);
    return true;
  }
  else
    return false;
} //End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void saveBestInd(vector<individual> &P, float time)
{ // fonction qui sauvegarde le meilleur individu plus elle historise ses performances et le temps d'exécution du processus d'évolution dans le fichier log.xls
  sort(P, 'c');
  int best = 0;
  for (int i = 0; i < P.size(); i++)
    if (P[best].MSEtest > P[i].MSEtest)
      best = i;

  string dos = "./save/";
  dos.append(dataname);
  string logpath = dos;
  ostringstream oss;
  oss << P[best].netParam[0].size();
  string result = oss.str();
  dos.append("/");
  dos.append(result);
  dos.append("-SavedWeights.xls");
  while (FileExists(dos.c_str()))
  {
    dos.insert(dos.end() - 17, '+');
  }
  ofstream file(dos.c_str());
  file << "individu:" << best << "\nfit=" << P[best].fit << "\ttrainMSE=" << P[best].MSEtrain << "\tvalidMSE=" << P[best].MSEvalid << "\ttestMSE=" << P[best].MSEtest << /*"\ttrainWTA="<<P[best].WTAtrain<<"\tvalidWTA="<<P[best].WTAvalid<<"\ttestWTA="<<P[best].WTAtest<<*/ "*\n";
  for (int j = 0; j < P[best].netParam.size(); j++)
    for (int l = 0; l < P[best].netParam[j].size(); l++)
    {
      file << "C" << j << ";";
      file << "N" << l << ";";
      file << P[best].netBiais[j][l] << ";";
      for (int m = 0; m < P[best].netParam[j][l].size(); m++)
        file << P[best].netParam[j][l][m] << ";";
      file << "\n";
    }
  file.close();
  result.clear();
  logpath.append("/log.xls");
  ofstream file1(logpath.c_str(), ios::app);
  //if (!FileExists("./save/log.xls")) file1<<"individu Num;Fit;trainMSE;validMSE;testMSE"<<endl;
  file1 << P[best].netParam[0].size() << ";" << time << ";" << P[best].fit << ";" << P[best].MSEtrain << ";" << P[best].MSEvalid << ";" << P[best].MSEtest << endl;
  file1.close();

  dos.clear();

} //End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void loadWeight(individual &P, string PHN)
{ // fonction de chargement des poids synaptiques et des biais poue le test d'une architecture existante
  string link, hn;

  string path = "./save/";
  path.append(dataname);
  //cout<<"\nDonner le lien du dossier d'ou les poids seront chargés :  ";cin>>hn;
  path.append("/");

  link = path;
  link.append(PHN);
  link.append("-SavedWeights.xls");
  ifstream file(link.c_str());
  if (file)
    cout << "\nL'ouverture du fichier est reussie ! :)" << endl;
  else
  {
    cerr << "\nImpossible d'ouvrir le fichier !!! :(" << endl;
    exit(1);
  }
  float tmp;
  file.ignore(20, '\n');
  file.ignore(200, '*');

  for (int j = 0; j < P.netParam.size(); j++)
    for (int k = 0; k < P.netParam[j].size(); k++)
    {
      file.ignore(4, ';');
      file.ignore(4, ';');
      file >> P.netBiais[j][k];
      file.ignore(1, ';');
      for (int l = 0; l < P.netParam[j][k].size(); l++)
      {
        file >> P.netParam[j][k][l];
        file.ignore(1, '\n');
      }
    }
  file.close();
} //End of function
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void importFile(vector<vector<float>> &inFin, vector<vector<float>> &outdFin, string link)
{ // fonction de lecture de bases de donées
  vector<vector<float>> matrix;
  vector<vector<float>> in;
  vector<vector<float>> outd;
  ifstream file(link.c_str());
  if (file)
    cout << "\nL'ouverture du fichier est reussie ! :)" << endl; // si l'ouverture a réussi
  else                                                           // sinon
  {
    cerr << "\nImpossible d'ouvrir le fichier !!! :(" << endl;
    exit(1);
  }
  float tmp;
  int n, p, x;
  file >> n;
  file >> p;
  file >> x;
  matrix.resize(n);
  for (int i = 0; i < matrix.size(); i++)
    matrix[i].resize(p);
  for (int i = 0; i < matrix.size(); i++)
    for (int j = 0; j < matrix[i].size(); j++)
      file >> matrix[i][j];
  file.close();
  outd.resize(n);
  in.resize(n);
  for (int i = 0; i < in.size(); i++)
    for (int j = 0; j < x; j++)
      in[i].push_back(matrix[i][j]);
  for (int i = 0; i < outd.size(); i++)
    for (int j = x; j < p; j++)
      outd[i].push_back(matrix[i][j]);
  /*for(int i=0;i<in.size();i++){
    for(int j=0;j<in[i].size();j++)
      cout<<in[i][j]<<" || ";
    cout<<endl;}
  for(int i=0;i<outd.size();i++){
    for(int j=0;j<outd[i].size();j++)
      cout<<outd[i][j]<<" || ";
    cout<<endl;}*/
  inFin = in;
  outdFin = outd;

  link.clear();
  matrix.clear();
  in.clear();
  outd.clear();
} //End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
string loadData()
{ //fonction qui importe et organise les données d'entainement, de validation et de test
  string trainfolder, validfolder, testfolder, tmp;
  cout << "Entrer le nom du dossier qui contient les données: ";
  cin >> tmp;
  testfolder = validfolder = trainfolder = tmp;
  trainfolder.append("/train.csv");
  validfolder.append("/valid.csv");
  testfolder.append("/test.csv");
  //cout<<trainF;
  importFile(trainIn, trainDo, trainfolder);
  importFile(validIn, validDo, validfolder);
  importFile(testIn, testDo, testfolder);
  return tmp;
} //End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
/*void MSEAlog(vector<individual> &P,int lignelogMSE){ // fonction qui enregistre les erreurs moyenne de la population à chaque génération
  ofstream logtrainMSEA("./graph/logtrainMSEA.xls", ios::app);
  ofstream logvalidMSEA("./graph/logvalidMSEA.xls", ios::app);
  ofstream logtestMSEA("./graph/logtestMSEA.xls", ios::app);
  logtrainMSEA<<lignelogMSE<<" "<<MSEAtrain(P)<<endl;
  logvalidMSEA<<lignelogMSE<<" "<<MSEAvalid(P)<<endl;
  logtestMSEA<<lignelogMSE<<" "<<MSEAtest(P)<<endl;
  logtrainMSEA.close();
  logvalidMSEA.close();
  logtestMSEA.close();
}*/
//End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
/*void NeurNblog(vector<individual> &P,string link,int Gen){// fonction qui enregistre le nombre de neurone atteint en temps réel
  string path="./graph/";
  link.append(dataname);
  link.append("_HLS.xls");
  path.append(link);
  ofstream NeurNblog(path.c_str(), ios::app);
  //int result=0;
  NeurNblog<<Gen<<" ";
  for(int k=0;k<P.size();k++)
  NeurNblog<<P[k].netParam[0].size()<<" ";
  NeurNblog<<endl;
  NeurNblog.close();
}*/
//End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void TotalMSEAlog(vector<individual> &P, vector<float> &MSEAlogTest, int Gen)
{ // fonction qui enregistre la variation de l'erreur du test durant les générations
  MSEAlogTest[Gen] = MSEAtest(P);
} //End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void evolve(vector<individual> &P, vector<float> &MSEAlogTest, int run)
{
  clock_t t;
  t = clock();

  initialize(P);
  fitness(P);
  vector<individual> oldP(P);
  //system("rm -fr ./save/*");
  //system("rm -fr ./graph/*");
  stop = totalGenerations;
  float OptValErr = MSEAvalid(P);
  do
  {
    Create_Mu_Lambda(P, mu, lambda, prodcutionRatio);
    NeurRecombination(lambda);
    NeurMutation(lambda);
    spMutation(lambda);
    opMutation(lambda);
    PopBP(trainIn, trainDo, lambda, probBP, etaBP, iterBP);
    fitness(lambda);
    //fitness(mu);
    oldP = P;
    selection(P, mu, lambda, 1.7, selectionType);
    OptValErr = OptValErr > MSEAvalid(P) ? MSEAvalid(P) : OptValErr;
    showpop(P);
    cout << "OptValErr=" << OptValErr << endl;
    cout << "GL(t)=" << (MSEAvalid(P) / OptValErr - 1) << endl;
    cout << "Run N°" << run + 1 << endl;
    //MSEAlog(P,totalGenerations-stop);
    //NeurNblog(lambda,"Produced_Individuals_",totalGenerations-stop);
    //NeurNblog(P,"Selected_Individuals_",totalGenerations-stop);
    TotalMSEAlog(P, MSEAlogTest, totalGenerations - stop);
    stop--;
    mu.clear();
    lambda.clear();
    //system("read x");
  } while (stop > 0 && eps > (MSEAvalid(P) / OptValErr - 1));
  //savepop(P);
  //system("bash scriptplot2.sh");
  t = clock() - t;
  float timeC = ((float)t) / CLOCKS_PER_SEC;
  saveBestInd(oldP, timeC);

} //End of function.
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void Construct(int runs)
{ // constructeur principale de l'algorithme d'évolution de PMC, il fait appel à toutes les fonctions déclarées ci-dessus
  dataname = loadData();
  system("mkdir ./save");
  string savefile = "mkdir ./save/";
  savefile.append(dataname);
  system(savefile.c_str());
  string wipefile = "rm -fr ./save/";
  wipefile.append(dataname);
  wipefile.append("/*");
  system(wipefile.c_str());
  vector<vector<individual>> Allpop(runs);
  vector<vector<float>> MSEAlogTest(runs);
  for (int i = 0; i < runs; i++)
    Allpop[i].resize(popsize);
  for (int i = 0; i < runs; i++)
    MSEAlogTest[i].resize(totalGenerations);
  //system("rm -fr ./graph/*");

  for (int i = 0; i < runs; i++)
  {
    srand(time(0));
    cout << "\nrun :" << i;
    evolve(Allpop[i], MSEAlogTest[i], i);
  }
  string path = "./graph/";
  path.append(dataname);
  path.append("_MSEA_test_log.xls");
  ofstream test10MSEA(path.c_str());
  for (int i = 0; i < totalGenerations; i++)
  {
    test10MSEA << i;
    for (int j = 0; j < runs; j++)
      test10MSEA << " " << MSEAlogTest[j][i];
    test10MSEA << endl;
  }

  //system("bash scriptplot.sh");
  //savepop(bestInd);
  Allpop.clear();
} //End of function
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void TestConstruct(int pop)
{ // constructeur secondaire de test des architectures sauvegardées qui évalue les architectures et re-enrigstre les résultats d'erreurs dans le fichier CEP_log.xls
  dataname = loadData();
  vector<individual> P(pop);
  vector<int> PHN(pop);
  vector<string> strPHN(pop);

  cout << "topologies:";

  int k = P.size();
  string path1 = "./save/";
  path1.append(dataname);
  path1.append("/log.xls");
  ifstream file1(path1.c_str());
  for (int i = 0; i < k; i++)
  {
    file1.ignore(1, ';');
    file1 >> PHN[i];
    file1.ignore(200, '\n');
    P[i].netParam.resize(2);
  }
  file1.close();

  for (int i = 0; i < k; i++)
  {
    ostringstream oss;
    oss << PHN[i];
    strPHN[i] = oss.str();
  }

  for (int i = 0; i < k - 1; i++)
    for (int j = i + 1; j < k; j++)
    {
      if (PHN[i] == PHN[j])
        strPHN[j].append("+");
    }

  for (int i = 0; i < k; i++)
  {
    P[i].netParam[0].resize(PHN[i]);
    P[i].netParam[1].resize(trainDo[0].size());
  }

  for (int i = 0; i < k; i++)
  {
    for (int j = 0; j < P[i].netParam.size(); j++)
      for (int l = 0; l < P[i].netParam[j].size(); l++)
        if (j == 0)
          P[i].netParam[j][l].resize(trainIn[0].size());
        else
          P[i].netParam[j][l].resize(P[i].netParam[j - 1].size());
  }
  for (int i = 0; i < k; i++)
  {
    P[i].netBiais.resize(P[i].netParam.size());
    for (int j = 0; j < P[i].netParam.size(); j++)
      P[i].netBiais[j].resize(P[i].netParam[j].size());
  }

  for (int i = 0; i < k; i++)
    loadWeight(P[i], strPHN[i]);
  fitness(P);

  string path = "./save/";
  path.append(dataname);
  path.append("/CEP_log.xls");
  ofstream file(path.c_str(), ios::app);
  for (int i = 0; i < P.size(); i++)
  {
    file << P[i].netParam[0].size() << ";fit:;" << P[i].fit << ";MSE:;" << P[i].MSEtrain << ";" << P[i].MSEvalid << ";" << P[i].MSEtest << ";WTA:;" << P[i].WTAtrain << ";" << P[i].WTAvalid << ";" << P[i].WTAtest << endl;
  }
  file.close();
} //End of function
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
