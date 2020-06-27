#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>

namespace py = pybind11;
using namespace pybind11::literals;
using npy = py::array_t<double, py::array::c_style | py::array::forcecast>;
using tuple = std::tuple<npy,npy,npy,npy,npy,npy,npy>;

double zeta (double x)
{
  const double halfpi = 1.5707963267948966;
  if (x < 0.0) return 1.0;
  else if (x > halfpi) return 0.0;
  else return cos(x);
}

double kronecker (double x, double y)
{
  if (x == y) return 1.0;
  else return 0.0;
}

tuple macula (npy T, npy Theta_star, npy Theta_spot, npy Theta_inst, npy Tstart, npy Tend,
                    bool derivatives = false, bool temporal = false, bool TdeltaV = false)
{
  // PRE-AMBLE ////////////////////////////////////////////////////////////////
  // Note - There should be no need to ever change these four parameters.
  const int pstar = 12;
  const int pspot = 8;
  const int pinst = 2;
  const int pLD = 5;
  // INPUTS ///////////////////////////////////////////////////////////////////
  auto t_ = T.request();
  auto theta_star_ = Theta_star.request();
  auto theta_spot_ = Theta_spot.request();
  auto theta_inst_ = Theta_inst.request();
  auto tstart_ = Tstart.request();
  auto tend_ = Tend.request();
  if (t_.ndim != 1) {
    throw std::runtime_error("t should be 1-D, shape (ndata,)");
  }
  if (theta_star_.ndim != 1) {
    throw std::runtime_error("theta_star should be 1-D, shape (12,)");
  }
  if (theta_star_.shape[0] != pstar) {
    throw std::runtime_error("Wrong number of star params (there should be 12)");
  }
  if (theta_spot_.ndim != 2) {
    throw std::runtime_error("theta_spot should be 2-D, shape (8, Nspot)");
  }
  if (theta_spot_.shape[0] != pspot) {
    throw std::runtime_error("Wrong number of spot params (there should be 8)");
  }
  if (theta_inst_.ndim != 2) {
    throw std::runtime_error("theta_inst should be 2-D, shape (2, mmax)");
  }
  if (theta_inst_.shape[0] != pinst) {
    throw std::runtime_error("Wrong number of inst params (there should be 2)");
  }
  if (tstart_.shape[0] != theta_inst_.shape[1]) {
    throw std::runtime_error("tstart should have shape (mmax,)");
  }
  if (tend_.shape[0] != theta_inst_.shape[1]) {
    throw std::runtime_error("tend should have shape (mmax,)");
  }
  double *t = (double *) t_.ptr,
    *theta_star = (double *) theta_star_.ptr,
    *theta_spot = (double *) theta_spot_.ptr,
    *theta_inst = (double *) theta_inst_.ptr,
    *tstart = (double *) tstart_.ptr,
    *tend = (double *) tend_.ptr;
  // VARIABLES ////////////////////////////////////////////////////////////////
  const int ndata = t_.shape[0];
  const int Nspot = theta_spot_.shape[1];
  const int mmax = theta_inst_.shape[1];
  const int jmax = pstar + pspot * Nspot + pinst * mmax;
  double tref[Nspot]; // By default, macula will set tref[k]=tmax[k]
  double SinInc, CosInc;
  const double pi = 3.141592653589793;
  const double halfpi = 1.5707963267948966;
  const double piI = 0.3183098861837907;
  const double tol = 0.0001; // alpha values below this will be ignored
  const double mingress = 0.0416667; // minimum ingress/egress time allowed
  // OUTPUTS //////////////////////////////////////////////////////////////////
  npy Fmod(ndata), Deltaratio(ndata);
  double *fmod = (double *) Fmod.request().ptr;
  double *deltaratio = (double *) Deltaratio.request().ptr;
  npy dFmoddt(ndata);
  double *dfmoddt = (double *) dFmoddt.request().ptr;
  npy dFmod_star(ndata * pstar);
  npy dFmod_spot(ndata * pspot * Nspot);
  npy dFmod_inst(ndata * pinst * mmax);
  double *dfmod_star = (double *) dFmod_star.request().ptr;
  double *dfmod_spot = (double *) dFmod_spot.request().ptr;
  double *dfmod_inst = (double *) dFmod_inst.request().ptr;
  
  /////////////////////////////////////////////////////////////////////////////
  //                       SECTION 1: THETA ASSIGNMENT                       //
  /////////////////////////////////////////////////////////////////////////////
  
  double Theta[jmax];

  int l = 0;
  for (int j = 0; j < pstar; ++j) {
    Theta[l++] = theta_star[j];
  }
  for (int k = 0; k < Nspot; ++k) {
    for (int j = 0; j < pspot; ++j) {
      Theta[l++] = theta_spot[j * Nspot + k];
    }
  }
  for (int j = 0; j < pinst; ++j) {
    for (int m = 0; m < mmax; ++m) {
      Theta[l++] = theta_inst[j * mmax + m];
    }
  }
  
  // Thus we have...
  // Theta_star[j] = Theta[j], 0 <= j < pstar
  // Theta_spot[j][k] = Theta[pstar + pspot * k + j], 0 <= j < pspot, 0 <= k < Nspot
  // Theta_inst[j][m] = Theta[pstar + pspot * Nspot + mmax * j + m], 0 <= j < pinst, 0 <= m < mmax

  /////////////////////////////////////////////////////////////////////////////
  //                       SECTION 2: BASIC PARAMETERS                       //
  /////////////////////////////////////////////////////////////////////////////
  
  double c[pLD], d[pLD];
  double U[mmax], B[mmax];
  double Box[mmax][ndata];
  double Phi0[Nspot], SinPhi0[Nspot], CosPhi0[Nspot], Prot[Nspot];
  double beta[Nspot][ndata], sinbeta[Nspot][ndata], cosbeta[Nspot][ndata];
  double alpha[Nspot][ndata], sinalpha[Nspot][ndata], cosalpha[Nspot][ndata];
  double Lambda[Nspot][ndata], sinLambda[Nspot][ndata], cosLambda[Nspot][ndata];
  double tcrit1[Nspot], tcrit2[Nspot], tcrit3[Nspot], tcrit4[Nspot];
  double alphamax[Nspot], fspot[Nspot], tmax[Nspot], life[Nspot], ingress[Nspot], egress[Nspot];

  // c and d assignment
  for (int n = 1; n < pLD; ++n) {
    c[n] = Theta[n+3];
    d[n] = Theta[n+7];
  }
  c[0] = 1.0 - c[1] - c[2] - c[3] - c[4]; // c0
  d[0] = 1.0 - d[1] - d[2] - d[3] - d[4]; // d0
  
  // inclination substitutions
  SinInc = sin(Theta[0]);
  CosInc = cos(Theta[0]);

  // U and B assignment
  for (int m = 0; m < mmax; ++m) {
    U[m] = Theta[pstar + pspot * Nspot + m];
    B[m] = Theta[pstar + pspot * Nspot + mmax + m];
  }

  // Box-car function (labelled as Pi_m in the paper)
  for (int i = 0; i < ndata; ++i) {
    for (int m = 0; m < mmax; ++m) {
      if (t[i] > tstart[m] && t[i] < tend[m])
        Box[m][i] = 1.0;
      else
        Box[m][i] = 0.0;
    }
  }
  
  // Spot params
  for (int k = 0; k < Nspot; ++k) {
    // Phi0 & Prot calculation
        Phi0[k] = Theta[pstar + pspot * k + 1];
    SinPhi0[k] = sin(Phi0[k]);
    CosPhi0[k] = cos(Phi0[k]);
    Prot[k] = Theta[1] / (1.0 - Theta[2] * pow(SinPhi0[k], 2) - Theta[3] * pow(SinPhi0[k], 4));
    // alpha calculation
    alphamax[k] = Theta[pstar + pspot * k + 2];
       fspot[k] = Theta[pstar + pspot * k + 3];
        tmax[k] = Theta[pstar + pspot * k + 4];
        life[k] = Theta[pstar + pspot * k + 5];
     ingress[k] = Theta[pstar + pspot * k + 6];
      egress[k] = Theta[pstar + pspot * k + 7];
    if (ingress[k] < mingress) { // minimum ingress time
      ingress[k] = mingress;
    }
    if (egress[k] < mingress) { // minimum egress time
      egress[k] = mingress;
    }
    // macula defines the reference time = maximum spot-size time
    // However, one can change the line below to whatever they wish.
    tref[k] = tmax[k];
    // tcrit points = critical instances in the evolution of the spot
    tcrit1[k] = tmax[k] - 0.5 * life[k] - ingress[k];
    tcrit2[k] = tmax[k] - 0.5 * life[k];
    tcrit3[k] = tmax[k] + 0.5 * life[k];
    tcrit4[k] = tmax[k] + 0.5 * life[k] + egress[k];
  }
  
  // alpha, lambda & beta
  for (int i = 0; i < ndata; ++i) {
    for (int k = 0; k < Nspot; ++k) {
      // temporal evolution of alpha
      if (t[i] < tcrit1[k] || t[i] > tcrit4[k])
        alpha[k][i] = 0.0;
      else if (t[i] < tcrit3[k] && t[i] > tcrit2[k])
        alpha[k][i] = alphamax[k];
      else if (t[i] <= tcrit2[k] && t[i] >= tcrit1[k])
        alpha[k][i] = alphamax[k] * ((t[i] - tcrit1[k]) / ingress[k]);
      else
        alpha[k][i] = alphamax[k] * ((tcrit4[k] - t[i]) / egress[k]);
      sinalpha[k][i] = sin(alpha[k][i]);
      cosalpha[k][i] = cos(alpha[k][i]);
      // Lambda & beta calculation
      Lambda[k][i] = Theta[pstar + pspot * k] + 2.0 * pi * (t[i] - tref[k]) / Prot[k];
      sinLambda[k][i] = sin(Lambda[k][i]);
      cosLambda[k][i] = cos(Lambda[k][i]);
      cosbeta[k][i] = CosInc * SinPhi0[k] + SinInc * CosPhi0[k] * cosLambda[k][i];
      beta[k][i] = acos(cosbeta[k][i]);
      sinbeta[k][i] = sin(beta[k][i]);
    }
  }
  
  /////////////////////////////////////////////////////////////////////////////
  //                        SECTION 3: COMPUTING FMOD                        //
  /////////////////////////////////////////////////////////////////////////////
  
  double zetaneg[Nspot][ndata], zetapos[Nspot][ndata];
  double Upsilon[pLD][Nspot][ndata], w[pLD][Nspot][ndata];
  double Psi[Nspot][ndata], Xi[Nspot][ndata];
  double q, A[Nspot][ndata];
  double Fab0;
  double Fab[ndata];

  // Fab0
  Fab0 = 0.0;
  for (int n = 0; n < pLD; ++n) {
    Fab0 += (n * c[n]) / (n + 4.0);
  }
  Fab0 = 1.0 - Fab0;

  // master fmod loop
  for (int i = 0; i < ndata; ++i) {
    Fab[i] = Fab0;
    fmod[i] = 0.0;
    for (int k = 0; k < Nspot; ++k) {
      // zetapos and zetaneg
      zetapos[k][i] = zeta(beta[k][i] + alpha[k][i]);
      zetaneg[k][i] = zeta(beta[k][i] - alpha[k][i]);
      // Area A
      if (alpha[k][i] > tol) {
        if (beta[k][i] > (halfpi + alpha[k][i])) {
          // Case IV
          A[k][i] = 0.0;
        }
        else if (beta[k][i] < (halfpi - alpha[k][i])) {
          // Case I
          A[k][i] = pi * cosbeta[k][i] * pow(sinalpha[k][i], 2);
        }
        else {
          // Case II & III
          Psi[k][i] = sqrt(1.0 - pow((cosalpha[k][i] / sinbeta[k][i]), 2));
          Xi[k][i] = sinalpha[k][i] * acos(-(cosalpha[k][i] * cosbeta[k][i])
                                          / (sinalpha[k][i] * sinbeta[k][i]));
          A[k][i] = acos(cosalpha[k][i] / sinbeta[k][i])
            + Xi[k][i] * cosbeta[k][i] * sinalpha[k][i]
            - Psi[k][i] * sinbeta[k][i] * cosalpha[k][i];
        }
      }
      else {
        A[k][i] = 0.0;
      }
      q = 0.0;
      // Upsilon & w
      for (int n = 0; n < pLD; ++n) {
        Upsilon[n][k][i] = pow(zetaneg[k][i], 2) - pow(zetapos[k][i], 2)
                            + kronecker(zetapos[k][i], zetaneg[k][i]);
        Upsilon[n][k][i] = (sqrt(pow(zetaneg[k][i], n+4))
                            - sqrt(pow(zetapos[k][i], n+4))) / Upsilon[n][k][i];
        w[n][k][i] = (4.0 * (c[n] - d[n] * fspot[k])) / (n + 4.0);
        w[n][k][i] *= Upsilon[n][k][i];
        // q
        q += (A[k][i] * piI) * w[n][k][i];
      }
      // Fab
      Fab[i] -= q;
    }
    for (int m = 0; m < mmax; ++m) {
      fmod[i] += U[m] * Box[m][i] * (Fab[i] / (Fab0 * B[m]) + (B[m] - 1.0) / B[m]);
    }
  }

  // delta {obs}/delta
  if (TdeltaV) {
    for (int i = 0; i < ndata; ++i) {
      deltaratio[i] = 0.0;
      for (int m = 0; m < mmax; ++m) {
        deltaratio[i] += B[m] * Box[m][i];
      }
      deltaratio[i] = (Fab0 / Fab[i]) / deltaratio[i];
    }
  }
  else {
    std::fill(deltaratio, deltaratio+ndata, 1.0);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  //                       SECTION 4: BASIS DERIVATIVES                      //
  /////////////////////////////////////////////////////////////////////////////
  
  double dc[pLD][jmax], dd[pLD][jmax];
  double dU[mmax][jmax], dB[mmax][jmax];
  double dfspot[Nspot][jmax];
  
  // Master if-loop
  if (derivatives) {
    // derivatives of c & d.
    for (int n = 0; n < pLD; ++n) {
      std::fill(dc[n], dc[n]+jmax, 0.0);
    }
    dc[1][4] = 1.0;
    dc[2][5] = 1.0;
    dc[3][6] = 1.0;
    dc[4][7] = 1.0;
    dc[0][4] = -1.0;
    dc[0][5] = -1.0;
    dc[0][6] = -1.0;
    dc[0][7] = -1.0;

    for (int n = 0; n < pLD; ++n) {
      std::fill(dd[n], dd[n]+jmax, 0.0);
    }
    dd[1][8] = 1.0;
    dd[2][9] = 1.0;
    dd[3][10] = 1.0;
    dd[4][11] = 1.0;
    dd[0][8] = -1.0;
    dd[0][9] = -1.0;
    dd[0][10] = -1.0;
    dd[0][11] = -1.0;

    // derivatives of U
    for (int m = 0; m < mmax; ++m) {
      std::fill(dU[m], dU[m]+jmax, 0.0);
      dU[m][pstar + pspot * Nspot + m] = 1.0;
    }

    // derivatives of B
    for (int m = 0; m < mmax; ++m) {
      std::fill(dB[m], dB[m]+jmax, 0.0);
      dB[m][pstar + pspot * Nspot + mmax + m] = 1.0;
    }

    // Derivatives of fspot (4th spot parameter)
    for (int k = 0; k < Nspot; ++k) {
      std::fill(dfspot[k], dfspot[k]+jmax, 0.0);
      dfspot[k][pstar + pspot * k + 3] = 1.0;
    }

    ///////////////////////////////////////////////////////////////////////////
    //                  SECTION 5: ALPHA & BETA DERIVATIVES                  //
    ///////////////////////////////////////////////////////////////////////////
    
    double dalpha[Nspot][ndata][jmax];
    double dbeta[Nspot][ndata][jmax];

    // Derivatives of alpha & beta
    for (int i = 0; i < ndata; ++i) {
      for (int k = 0; k < Nspot; ++k) {
        // Derivatives of alpha(alphamax,tmax,life,ingress,egress)
        // [function of 5*Nspot parameters]
        std::fill(dalpha[k][i], dalpha[k][i]+jmax, 0.0);
        // wrt alphamax (3rd spot parameter)
        dalpha[k][i][pstar + pspot * k + 2] = alpha[k][i] / alphamax[k];
        // wrt tmax (5th spot parameter)
        if (t[i] < tcrit2[k] && t[i] > tcrit1[k])
          dalpha[k][i][pstar + pspot * k + 4] = -alphamax[k] / ingress[k];
        else if (t[i] < tcrit4[k] && t[i] > tcrit3[k])
          dalpha[k][i][pstar + pspot * k + 4] = alphamax[k] / egress[k];
        // wrt life (6th spot parameter)
        if (t[i] < tcrit2[k] && t[i] > tcrit1[k])
          dalpha[k][i][pstar + pspot * k + 5] = 0.5 * alphamax[k] / ingress[k];
        else if (t[i] < tcrit4[k] && t[i] > tcrit3[k])
          dalpha[k][i][pstar + pspot * k + 5] = 0.5 * alphamax[k] / egress[k];
        // wrt ingress (7th spot parameter)
        if (t[i] < tcrit2[k] && t[i] > tcrit1[k]) {
          dalpha[k][i][pstar + pspot * k + 6] = -(alphamax[k] / pow(ingress[k], 2)) *
            (t[i] - 0.50 * (tcrit1[k] + tcrit2[k]));
        }
        // wrt egress (8th spot parameter)
        if (t[i] < tcrit4[k] && t[i] > tcrit3[k]) {
          dalpha[k][i][pstar + pspot * k + 7] = (alphamax[k] / pow(egress[k], 2)) *
            (t[i] - 0.50 * (tcrit3[k] + tcrit4[k]));
        }
        // Stellar derivatives of beta(Istar,Phi0,Lambda0,Peq,kappa2,kappa4)
        // [Function of 4+2*Nspot parameters]
        std::fill(dbeta[k][i], dbeta[k][i]+jmax, 0.0);
        // wrt Istar (1st star parameter)
        dbeta[k][i][0] = SinPhi0[k] * SinInc - cosLambda[k][i] * CosPhi0[k] * CosInc;
        dbeta[k][i][0] /= sinbeta[k][i];
        // wrt Peq (2nd star parameter)
        dbeta[k][i][1] = CosPhi0[k] * sinLambda[k][i] * SinInc / sinbeta[k][i];
        dbeta[k][i][1] *= 2.0 * pi * (t[i] - tref[k]) / Theta[1]; // Temporary
        // wrt kappa2 (3rd star parameter)
        dbeta[k][i][2] = -dbeta[k][i][1] * pow(SinPhi0[k], 2);
        // wrt kappa4 (4th star parameter)
        dbeta[k][i][3] = -dbeta[k][i][1] * pow(SinPhi0[k], 4);
        // wrt Peq continued
        dbeta[k][i][1] = -dbeta[k][i][1] / Prot[k];
        // Spot-derivatives of beta
        // wrt Lambda [1st spot parameter]
        dbeta[k][i][pstar + pspot * k] = SinInc * CosPhi0[k] * sinLambda[k][i] / sinbeta[k][i];
        // wrt Phi0 [2nd spot parameter]
        dbeta[k][i][pstar + pspot * k + 1] = 2.0 * Theta[2] * pow(CosPhi0[k], 2)
          + Theta[3] * pow(2.0 * SinPhi0[k] * CosPhi0[k], 2);
        dbeta[k][i][pstar + pspot * k + 1] *= 2.0 * pi * (t[i] - tref[k]) / Theta[1];
        dbeta[k][i][pstar + pspot * k + 1] = cosLambda[k][i] - dbeta[k][i][pstar + pspot * k + 1];
        dbeta[k][i][pstar + pspot * k + 1] *= SinInc * SinPhi0[k] / sinbeta[k][i];
      }
    }

    ///////////////////////////////////////////////////////////////////////////
    //                        SECTION 6: A DERIVATIVES                       //
    ///////////////////////////////////////////////////////////////////////////
    
    double epsil, dAda[Nspot][ndata], dAdb[Nspot][ndata];

    // Semi-derivatives of A
    for (int i = 0; i < ndata; ++i) {
      for (int k = 0; k < Nspot; ++k) {
        if (alpha[k][i] > tol) {
          if (beta[k][i] > (halfpi + alpha[k][i])) {
            // Case IV
            dAda[k][i] = 0.0;
            dAdb[k][i] = 0.0;
          }
          else if (beta[k][i] < (halfpi - alpha[k][i])) {
            // Case I
            dAda[k][i] = 2.0 * pi * cosbeta[k][i] * sinalpha[k][i] * cosalpha[k][i];
            dAdb[k][i] = -pi * pow(sinalpha[k][i], 2) * sinbeta[k][i];
          }
          else {
            // Case II & III
            epsil = 2.0 * (pow(cosalpha[k][i], 2) + pow(cosbeta[k][i], 2) - 1.0) /
                    (pow(sinbeta[k][i], 2) * Psi[k][i]);
            dAda[k][i] = -sinalpha[k][i] * sinbeta[k][i] * epsil
                        + 2.0 * cosalpha[k][i] * cosbeta[k][i] * Xi[k][i];
            dAdb[k][i] = 0.5 * cosalpha[k][i] * cosbeta[k][i] * epsil
                        - sinalpha[k][i] * sinbeta[k][i] * Xi[k][i];
          }
        }
        else {
          dAda[k][i] = 0.0;
          dAdb[k][i] = 0.0;
        }
      }
    }

    ///////////////////////////////////////////////////////////////////////////
    //                      SECTION 7: FINAL DERIVATIVES                     //
    ///////////////////////////////////////////////////////////////////////////
        
    double dA;
    double dUpsilon, dw;
    double dzetaneg;
    double dzetapos;
    double dq;
    double dFtilde;
    double dFab, dFmod[ndata][jmax];
    double dzetanegda[Nspot][ndata], dzetaposda[Nspot][ndata];
    double dFab0[jmax];

    // Derivatives of Fab0
    std::fill(dFab0, dFab0+jmax, 0.0);
    dFab0[4] = -0.2;
    dFab0[5] = -0.33333333;
    dFab0[6] = -0.42857143;
    dFab0[7] = -0.5;

    // derivatives main loop
    for (int j = 0; j < jmax; ++j) {
      for (int i = 0; i < ndata; ++i) {
        dFab = dFab0[j];
        dFmod[i][j] = 0.0;
        for (int k = 0; k < Nspot; ++k) {
          // Derivatives of A
          dA = dAda[k][i] * dalpha[k][i][j] + dAdb[k][i] * dbeta[k][i][j];
          // Derivatives of zeta wrt alpha (and implicitly beta)
          // dzetanegda
          if ((beta[k][i] - alpha[k][i]) < halfpi && (beta[k][i] - alpha[k][i]) > 0.0)
            dzetanegda[k][i] = cosalpha[k][i] * sinbeta[k][i] - cosbeta[k][i] * sinalpha[k][i];
          else
            dzetanegda[k][i] = 0.0;
          // dzetaposda
          if ((beta[k][i] + alpha[k][i]) < halfpi && (beta[k][i] + alpha[k][i]) > 0.0)
            dzetaposda[k][i] = -cosalpha[k][i] * sinbeta[k][i] - cosbeta[k][i] * sinalpha[k][i];
          else
            dzetaposda[k][i] = 0.0;
          // Derivatives of zeta
          dzetaneg = dzetanegda[k][i] * (dalpha[k][i][j] - dbeta[k][i][j]);
          dzetapos = dzetaposda[k][i] * (dalpha[k][i][j] + dbeta[k][i][j]);
          dq = 0.0;
          // Derivatives of Upsilon
          for (int n = 0; n < pLD; ++n) {
            dUpsilon = sqrt(pow(zetaneg[k][i], n+2)) * dzetaneg
              - sqrt(pow(zetapos[k][i], n+2)) * dzetapos;
            dUpsilon = 0.5 * (n+4.0) * dUpsilon - 2.0 * Upsilon[n][k][i] \
                        * (dzetaneg - dzetapos);
            dUpsilon /= (pow(zetaneg[k][i], 2) - pow(zetapos[k][i], 2)
                                     + kronecker(zetapos[k][i], zetaneg[k][i]));
            // Derivatives of w
            dw = Upsilon[n][k][i] * dc[n][j] + (c[n] - d[n] * fspot[k]) * dUpsilon -
                  d[n] * Upsilon[n][k][i] * dfspot[k][j] - 
                  fspot[k] * Upsilon[n][k][i] * dd[n][j];
            dw *= 4.0 / (n + 4.0);
            // Derivatives of q
            dq += (A[k][i] * dw + dA * w[n][k][i]) * piI;
          }
          dFab -= dq;
        }
        for (int m = 0; m < mmax; ++m) {
          dFtilde = Fab0 * B[m] * (Fab[i] + Fab0 * (B[m] - 1.0)) * dU[m][j]
                    + U[m] * (B[m] * Fab0 * dFab - B[m] * Fab[i] * dFab0[j]
                    + Fab0 * (Fab0 - Fab[i]) * dB[m][j]);
          dFtilde *= Box[m][i] / pow(Fab0*B[m], 2);
          dFmod[i][j] += dFtilde;
        }
      }
    }

    ///////////////////////////////////////////////////////////////////////////
    //                    SECTION 8: TEMPORAL DERIVATIVES                    //
    ///////////////////////////////////////////////////////////////////////////

    double dalphadt, dbetadt, dzetanegdt, dzetaposdt;
    double dqdt, dAdt;
    double dwdt, dUpsilondt;
    double dFtildedt;
    double dFabdt;
    
    if (temporal) {
      // Temporal derivatives of alpha and beta
      for (int i = 0; i < ndata; ++i) {
        dFabdt = 0.0;
        dfmoddt[i] = 0.0;
        for (int k = 0; k < Nspot; ++k) {
          if (t[i] < tcrit2[k] && t[i] > tcrit1[k])
            dalphadt = alphamax[k] / ingress[k];
          else if (t[i] < tcrit4[k] && t[i] > tcrit3[k])
            dalphadt = -alphamax[k] / egress[k];
          else
            dalphadt = 0.0;
          dbetadt = 1.0 - pow(SinInc * CosPhi0[k] * cosLambda[k][i] 
                              + CosInc * SinPhi0[k], 2);
          dbetadt = (2.0 * pi * SinInc * CosPhi0[k] * sinLambda[k][i]) /
                      (Prot[k] * sqrt(dbetadt));
          // Temporal derivatives of zeta
          dzetanegdt = dzetanegda[k][i] * (dalphadt - dbetadt);
          dzetaposdt = dzetaposda[k][i] * (dalphadt + dbetadt);
          // Temporal derivatives of A
          dAdt = dAda[k][i] * dalphadt + dAdb[k][i] * dbetadt;
          dqdt = 0.0;
          // Temporal derivatives of Upsilon
          for (int n = 0; n < pLD; ++n) {
            dUpsilondt = sqrt(pow(zetaneg[k][i], n+2)) * dzetanegdt
              - sqrt(pow(zetapos[k][i], n+2)) * dzetaposdt;
            dUpsilondt = 0.5 * (n + 4.0) * dUpsilondt
              - 2.0 * Upsilon[n][k][i] *
              (dzetanegdt - dzetaposdt);
            dUpsilondt /= (pow(zetaneg[k][i], 2) - pow(zetapos[k][i], 2)
                           + kronecker(zetapos[k][i], zetaneg[k][i]));
            // Temporal derivatives of w
            dwdt = dUpsilondt * ((4.0 * (c[n] - d[n] * fspot[k])) / (n + 4.0));
            // Temporal derivatives of q
            dqdt += (A[k][i] * dwdt + dAdt * w[n][k][i]) * piI;
          }
          // Temporal derivatives of Fab
          dFabdt -= dqdt;
        }
        // Temporal derivatives of Ftilde
        for (int m = 0; m < mmax; ++m) {
          dFtildedt = ((U[m] * Box[m][i]) / (B[m] * Fab0)) * dFabdt;
          // Temporal derivatives of Fmod
          dfmoddt[i] += dFtildedt;
        }
      }
    }
    else {
      std::fill(dfmoddt, dfmoddt+ndata, 0.0);
    }

    ///////////////////////////////////////////////////////////////////////////
    //                    SECTION 9: RE-SPLIT DERIVATIVES                    //
    ///////////////////////////////////////////////////////////////////////////

    // Derivatives provided for Theta_star, Theta_inst, Theta_spot discretely
    l = 0;
    for (int j = 0; j < pstar; ++j) {
      for (int i = 0; i < ndata; ++i) {
        dfmod_star[i * pstar + j] = dFmod[i][l];
      }
      l++;
    }
    for (int k = 0; k < Nspot; ++k) {
      for (int j = 0; j < pspot; ++j) {
        for (int i = 0; i < ndata; ++i) {
          dfmod_spot[i * pspot * Nspot + j * Nspot + k] = dFmod[i][l];
        }
        l++;
      }
    }
    for (int m = 0; m < mmax; ++m) {
      for (int j = 0; j < pinst; ++j) {
        for (int i = 0; i < ndata; ++i) {
          dfmod_inst[i * pinst * mmax + j * mmax + m] = dFmod[i][l];
        }
        l++;
      }
    }
  }
  else {
    std::fill(dfmod_star, dfmod_star+pstar*ndata, 0.0);
    std::fill(dfmod_spot, dfmod_spot+pspot*Nspot*ndata, 0.0);
    std::fill(dfmod_inst, dfmod_inst+pinst*mmax*ndata, 0.0);
    std::fill(dfmoddt, dfmoddt+ndata, 0.0);
  }

  // reshape output arrays
  dFmod_star.resize({ndata, pstar});
  dFmod_spot.resize({ndata, pspot, Nspot});
  dFmod_inst.resize({ndata, pinst, mmax});

  return std::make_tuple(Fmod, dFmoddt, dFmod_star, dFmod_spot, 
                         dFmod_spot, dFmod_inst, Deltaratio);
}

PYBIND11_MODULE(macula, m)
{
    m.doc() = "";
    m.def("macula", &macula, "",
          "t"_a, "theta_star"_a, "theta_spot"_a, "theta_inst"_a, "tstart"_a, "tend"_a,
          "derivatives"_a=false, "temporal"_a=false, "tdeltav"_a=false);
}
