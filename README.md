# SourceDependency2021
 repository for codes necessary to reproduce figures and analyis in Goodwell and Bassiouni, 2021


Binary Model Example (Matlab):
ModelExample1_binarymodels.m - This code will generate binary source/target distributions and reproduce Figure 2 from the manuscript


Synthetic/Generated Model Example (Matlab) :
ModelExample2_synthetic_models.m - This code will generate test source variables with different source dependencies, and perform weighted average, multiplication, and proportion operations as described in the paper.  This code will reproduce Figure 3 in the manuscript.

Weather Station Regression Model Example (Ta, WS, RH at 1-min temporal resolution, Matlab)
WeatherRegressions_1_Create.m - Run this code, with appropriate functions (noted below) to load dataset and train various types of models.  This code produces a .mat file called “modcollect.mat” with all results saved.  This data file is then loaded into the next code

WeatherRegressions_2_ITAnalysis.m - This code takes modcollect.mat as input, and a choice between reproducing main text model analysis (9 models) or additional models presented in SI (6 models, 4 machine learning models).  Data are partitioned into 5 day windows and IT measures are calculated for all models and observed data.  This code saves a file that is then input into the next code.

WeatherRegressions_3_Plots.m - This code takes as input the results file from WeatherRegressions_2_ITAnalysis.m  and reproduces Figures 4 and 5 from the manuscript, or Figure 1 from the Supplementary information.

Functions (Matlab):
trainRegression…m files - These are functions to train different types of regression models, generated from the regressionLearner tool in Matlab and then saved to train several model types based on weather station data.  For example, the trainRegressionModelLin1source.m generates a linear regression model based on a single source variable, which in the paper includes Ta, WS, Ta+WS, Ta-WS, Ta*WS and Ta/WS as the options of individual sources. The other models take two source variables as inputs for training.

ButterFiltFun.m - This function is used in the weather station regressions to filter the diurnal cycle from the Ta and RH 1min datasets.

compute_pdf.m - This function computes a 1D-3D probability distribution function (pdf) from input data, with N number of fixed bins.  This function assumes that the input data have already been normalized to a 0-1 range.

compute_info_measures.m - This functions takes a pdf as input, and computes relevant information theory measures, such as total information, S, U1, U2, and R, in a structure that is output.

Datasets (Matlab)
SFP2_AllData.mat - This data file contains all the data from the Sangamon River Forest Preserve, first applied in Goodwell and Kumar, WRR 2017a,b.  In this study, it is used in the weather station regression examples.

traindata.mat - This is a randomly generated training dataset (3 datasets, the first was used for the paper) from the SFP weather station data.  This is used in the code WeatherRegressions_1_Create.m to train all models, and can be randomly regenerated using that code.

modcollect.mat - This file is generated when the code WeatherRegressions_1_Create.m is run, and saves the workspace from that code.

Results082021_mainmodels.mat and Results082021_SImodels.mat: These are examples of results files that can be generated from running WeatherRegressions_2_ITAnalysis.m.  They can be input directly into WeatherRegressions_3_Plots.m to generate figures.

tight_subplot.m - obtained from https://www.mathworks.com/matlabcentral/fileexchange/27991-tight_subplot-nh-nw-gap-marg_h-marg_w to generate tight subplots to reproduce Figure 4 in the main text.

Ecohydrologic Model Example (SFE method vs SUMMA configurations, Python codes)
(to be added)