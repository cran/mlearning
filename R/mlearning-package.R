#' Machine Learning Algorithms with Unified Interface and Confusion Matrices
#'
#' @description
#' This package provides wrappers around several existing machine learning
#' algorithms in R, under a unified user interface. Confusion matrices can also
#' be calculated and viewed as tables or plots. Key features are:
#'
#' - Unified, formula-based interface for all algorithms, similar to
#' [stats::lm()].
#'
#' - Optimized code when a simplified formula `y ~ .` is used, meaning all
#' variables in data are used (one of them (`y` here) is the class to be
#' predicted (classification problem, a factor variable), or the dependent
#' variable of the model (regression problem, a numeric variable).
#'
#' - Similar way of dealing with missing data, both in the training set and in
#' predictions. Underlying algorithms deal differently with missing data. Some
#' accept them, other not.
#'
#' - Unified way of dealing with factor levels that have no cases in the
#' training set. The training succeeds, but the classifier is, of course, unable
#' to classify items in the missing class.
#'
#' - The [predict()] methods have similar arguments. They return the class,
#' membership to the classes, both, or something else (probabilities,
#' raw predictions, ...) depending on the algorithm or the problem
#' (classification or regression).
#'
#' - The [cvpredict()] method is available for all algorithms and it performs
#' very easily a cross-validation, or even a leave_one_out validation (when
#' `cv.k` = number of cases). It operates transparently for the end-user.
#'
#' - The [confusion()] method creates a confusion matrix and the object can be
#' printed, summarized, plotted. Various metrics are easily derived from the
#' confusion matrix. Also, it allows to adjust prior probabilities of the
#' classes in a classification problem, in order to obtain more representative
#' estimates of the metrics when priors are adjusted to values closes to real
#' proportions of classes in the data.
#'
#' See [mlearning()] for further explanations and an example analysis. See
#' [mlLda()] for examples of the different forms of the formula that can be
#' used. See [plot.confusion()] for the different ways to explore the confusion
#' matrix.

#' @section Important functions:
#'
#' - [ml_lda()], [ml_qda()], [ml_naive_bayes()], [ml_knn()], [ml_lvq()],
#' [ml_nnet()], [ml_rpart()], [ml_rforest()] and [ml_svm()] to train classifiers
#' or regressors with the different algorithms that are supported in the
#' package,
#'
#' - [predict()] and [cvpredict()] for predictions, including using
#' cross-validation,
#'
#' - [confusion()] to calculate the confusion matrix (with various methods to
#' analyze it and to calculate derived metrics like recall, precision, F-score,
#' ...)
#'
#'- [prior()] to adjust prior probabilities,
#'
#'- [response()] and [train()] to extract response and training variables from
#' an **mlearning** object.
#'
#' @docType package
#' @name mlearning-package

## usethis namespace: start
#' @importFrom graphics abline axis barplot image legend lines mtext par plot
#'   rect stars text
#' @importFrom stats addmargins hclust model.frame model.weights na.exclude
#'   na.fail na.omit na.pass predict terms
#' @importFrom grDevices cm.colors colorRampPalette hsv topo.colors
#'   terrain.colors
#' @importFrom class knn olvq1 lvq1 lvq2 lvq3 lvqinit lvqtest
#' @importFrom nnet nnet.default class.ind
#' @importFrom MASS lda qda
#' @importFrom e1071 naiveBayes svm
#' @importFrom randomForest randomForest
#' @importFrom ipred predict.ipredknn errorest control.errorest
#' @importFrom rpart rpart
## usethis namespace: end
NULL
