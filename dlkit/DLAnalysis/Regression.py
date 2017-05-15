# Analysis
from ROOT import TH1F, TCanvas, TF1
import numpy as np


def RegressionAnalysis(ModelH, X_test, y_test, M_min, M_max, BatchSize, M, V):
    print "Perdiction Analysis."
    result = ModelH.Model.predict(X_test, batch_size=BatchSize)
    MassNorm = (M_max - M_min)
    #    result=result*MassNorm-M_min
    result = result * V + M

    c1 = TCanvas("c1")

    resultHist = TH1F("Result", "Result", 100, 1.1 * M_min, 1.1 * M_max)
    map(lambda (x): resultHist.Fill(x), result.flatten())
    resultHist.Draw()
    c1.Print(ModelH.OutDir + "/Result.pdf")

    targetHist = TH1F("Target", "Target", 100, 1.1 * M_min, 1.1 * M_max)
    map(lambda (x): targetHist.Fill(x), y_test)
    targetHist.Draw()
    c1.Print(ModelH.OutDir + "/Target.pdf")

    residual = result.flatten() - y_test

    residualHist = TH1F("Residual", "Residual", 100, -5, 5)
    map(lambda (x): residualHist.Fill(x), residual)
    residualHist.Fit("gaus")

    fit = residualHist.GetFunction("gaus")
    chi2 = fit.GetChisquare()
    A = fit.GetParameter(0)
    A_sigma = fit.GetParError(0)

    mean = fit.GetParameter(1)
    mean_sigma = fit.GetParError(1)

    sigma = fit.GetParameter(2)
    sigma_sigma = fit.GetParError(2)

    ModelH.MetaData["ResidualMean"] = residualHist.GetMean()
    ModelH.MetaData["ResidualStdDev"] = residualHist.GetStdDev()

    ModelH.MetaData["ResidualFitChi2"] = chi2
    ModelH.MetaData["ResidualFitMean"] = [mean, mean_sigma]
    ModelH.MetaData["ResidualFitSigma"] = [sigma, sigma_sigma]

    residualHist.Draw()
    c1.Print(ModelH.OutDir + "/Residual.pdf")

    G1 = TF1("G1", "gaus", -10.0, 10.)
    G2 = TF1("G2", "gaus", -10., 10.0)

    residualHist.Fit(G1, "R", "", -10., 1.)
    residualHist.Fit(G2, "R", "", -1, 10.)

    DoubleG = TF1("DoubleG", "gaus(0)+gaus(3)", -10.0, 10.0);

    DoubleG.SetParameter(0, G1.GetParameter(0))
    DoubleG.SetParameter(1, G1.GetParameter(1))
    DoubleG.SetParameter(2, G1.GetParameter(2))

    DoubleG.SetParameter(3, G2.GetParameter(0))
    DoubleG.SetParameter(4, G2.GetParameter(1))
    DoubleG.SetParameter(5, G2.GetParameter(2))

    residualHist.Fit(DoubleG)

    fitres = []

    for ii in xrange(0, 6):
        fitres += [[DoubleG.GetParameter(ii), DoubleG.GetParError(ii)]]

    ModelH.MetaData["DoubleGaussianFit"] = fitres

    residualHist.Draw()
    c1.Print(ModelH.OutDir + "/Residual_2GFit.pdf")


def ClassificationAnalysis(ModelH, X_test, y_test, y_testT, M_min, M_max, NBins, BatchSize):
    print "Perdiction Analysis."

    resultClass = ModelH.Model.predict(X_test, batch_size=BatchSize)

    binwidth = (M_max - M_min) / NBins
    result = np.argmax(resultClass, axis=1) * binwidth + M_min

    c1 = TCanvas("c1")

    resultHist = TH1F("Result", "Result", NBins, M_min, M_max)
    map(lambda (x): resultHist.Fill(x), result.flatten())
    resultHist.Draw()
    c1.Print(ModelH.OutDir + "/Result.pdf")

    targetHist = TH1F("Target", "Target", NBins, M_min, M_max)
    map(lambda (x): targetHist.Fill(x), y_testT)
    targetHist.Draw()
    c1.Print(ModelH.OutDir + "/Target.pdf")

    residual = result.flatten() - y_testT

    residualHist = TH1F("Residual", "Residual", NBins, M_min, M_max)
    map(lambda (x): residualHist.Fill(x), residual)
    residualHist.Fit("gaus")

    fit = residualHist.GetFunction("gaus")
    chi2 = fit.GetChisquare()
    A = fit.GetParameter(0)
    A_sigma = fit.GetParError(0)

    mean = fit.GetParameter(1)
    mean_sigma = fit.GetParError(1)

    sigma = fit.GetParameter(2)
    sigma_sigma = fit.GetParError(2)

    ModelH.MetaData["ResidualMean"] = residualHist.GetMean()
    ModelH.MetaData["ResidualStdDev"] = residualHist.GetStdDev()

    ModelH.MetaData["ResidualFitChi2"] = chi2
    ModelH.MetaData["ResidualFitMean"] = [mean, mean_sigma]
    ModelH.MetaData["ResidualFitSigma"] = [sigma, sigma_sigma]

    residualHist.Draw()
    c1.Print(ModelH.OutDir + "/Residual.pdf")

    G1 = TF1("G1", "gaus", -10.0, 10.)
    G2 = TF1("G2", "gaus", -10., 10.0)

    residualHist.Fit(G1, "R", "", -10., 1.)
    residualHist.Fit(G2, "R", "", -1, 10.)

    DoubleG = TF1("DoubleG", "gaus(0)+gaus(3)", -10.0, 10.0);

    DoubleG.SetParameter(0, G1.GetParameter(0))
    DoubleG.SetParameter(1, G1.GetParameter(1))
    DoubleG.SetParameter(2, G1.GetParameter(2))

    DoubleG.SetParameter(3, G2.GetParameter(0))
    DoubleG.SetParameter(4, G2.GetParameter(1))
    DoubleG.SetParameter(5, G2.GetParameter(2))

    residualHist.Fit(DoubleG)

    fitres = []

    for ii in xrange(0, 6):
        fitres += [[DoubleG.GetParameter(ii), DoubleG.GetParError(ii)]]

    ModelH.MetaData["DoubleGaussianFit"] = fitres

    residualHist.Draw()
    c1.Print(ModelH.OutDir + "/Residual_2GFit.pdf")
    return result, resultClass
