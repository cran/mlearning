#' Supervised classification and regression using random forest
#'
#' @description
#' Unified (formula-based) interface version of the random forest algorithm
#' provided by [randomForest::randomForest()].
#'
#' @param formula a formula with left term being the factor variable to predict
#' (for supervised classification), a vector of numbers (for regression) or
#' nothing (for unsupervised classification) and the right term with the list
#' of independent, predictive variables, separated with a plus sign. If the
#' data frame provided contains only the dependent and independent variables,
#' one can use the `class ~ .` short version (that one is strongly encouraged).
#' Variables with minus sign are eliminated. Calculations on variables are
#' possible according to usual formula convention (possibly protected by using
#' `I()`).
#' @param data a data.frame to use as a training set.
#' @param train a matrix or data frame with predictors.
#' @param response a vector of factor (classification) or numeric (regression),
#'   or `NULL` (unsupervised classification).
#' @param ntree the number of trees to generate (use a value large enough to get
#'   at least a few predictions for each input row). Default is 500 trees.
#' @param mtry number of variables randomly sampled as candidates at each split.
#'   Note that the default values are different for classification (sqrt(p)
#'   where p is number of variables in x) and regression (p/3)?
#' @param replace sample cases with or without replacement (`TRUE` by default)?
#' @param classwt priors of the classes. Need not add up to one. Ignored for
#'   regression.
#' @param ... further arguments passed to [randomForest::randomForest()] or its
#'   [predict()] method. There are many more arguments, see the corresponding
#'   help page.
#' @param subset index vector with the cases to define the training set in use
#'   (this argument must be named, if provided).
#' @param na.action function to specify the action to be taken if `NA`s are
#'   found. For [ml_rforest()] `na.fail` is used by default. The calculation is
#'   stopped if there is any `NA` in the data. Another option is `na.omit`,
#'   where cases with missing values on any required variable are dropped (this
#'   argument must be named, if provided). For the `predict()` method, the
#'   default, and most suitable option, is `na.exclude`. In that case, rows with
#'   `NA`s in `newdata=` are excluded from prediction, but reinjected in the
#'   final results so that the number of items is still the same (and in the
#'   same order as `newdata=`).
#' @param object an **mlRforest** object
#' @param newdata a new dataset with same conformation as the training set (same
#'   variables, except may by the class for classification or dependent variable
#'   for regression). Usually a test set, or a new dataset to be predicted.
#' @param type the type of prediction to return. `"class"` by default, the
#'   predicted classes. Other options are `"membership"` the membership (number
#'   between 0 and 1) to the different classes as assessed by the number of
#'   neighbors of these classes, or `"both"` to return classes and memberships.
#'   One can also use `"vote"`, which returns the number of trees that voted
#'   for each class.
#' @param method `"direct"` (default), `"oob"` or `"cv"`. `"direct"` predicts
#'   new cases in `newdata=` if this argument is provided, or the cases in the
#'   training set if not. Take care that not providing `newdata=` means that you
#'   just calculate the **self-consistency** of the classifier but cannot use
#'   the metrics derived from these results for the assessment of its
#'   performances (in the case of Random Forest, these metrics would most
#'   certainly falsely indicate a perfect classifier). Either use a different
#'   data set in `newdata=` or use the alternate approaches: out-of-bag
#'   (`"oob"`) or cross-validation ("cv"). The out-of-bag approach uses
#'   individuals that are not used to build the trees to assess performances. It
#'   is an unbiased estimates. If you specify `method = "cv"` then [cvpredict()]
#'   is used and you cannot provide `newdata=` in that case.
#'
#' @return [ml_rforest()]/[mlRforest()] creates an **mlRforest**, **mlearning**
#'   object containing the classifier and a lot of additional metadata used by
#'   the functions and methods you can apply to it like [predict()] or
#'   [cvpredict()]. In case you want to program new functions or extract
#'   specific components, inspect the "unclassed" object using [unclass()].
#' @seealso [mlearning()], [cvpredict()], [confusion()], also
#'   [randomForest::randomForest()] that actually does the classification.
#' @export
#'
#' @examples
#' # Prepare data: split into training set (2/3) and test set (1/3)
#' data("iris", package = "datasets")
#' train <- c(1:34, 51:83, 101:133)
#' iris_train <- iris[train, ]
#' iris_test <- iris[-train, ]
#' # One case with missing data in train set, and another case in test set
#' iris_train[1, 1] <- NA
#' iris_test[25, 2] <- NA
#'
#' iris_rf <- ml_rforest(data = iris_train, Species ~ .)
#' summary(iris_rf)
#' plot(iris_rf) # Useful to look at the effect of ntree=
#' # For such a relatively simple case, 50 trees are enough
#' iris_rf <- ml_rforest(data = iris_train, Species ~ ., ntree = 50)
#' summary(iris_rf)
#' predict(iris_rf) # Default type is class
#' predict(iris_rf, type = "membership")
#' predict(iris_rf, type = "both")
#' predict(iris_rf, type = "vote")
#' # Out-of-bag prediction (unbiased)
#' predict(iris_rf, method = "oob")
#' # Self-consistency (always very high for random forest, biased, do not use!)
#' confusion(iris_rf)
#' # This one is better
#' confusion(iris_rf, method = "oob") # Out-of-bag performances
#' # Cross-validation prediction is also a good choice when there is no test set
#' predict(iris_rf, method = "cv")  # Idem: cvpredict(res)
#' # Cross-validation for performances estimation
#' confusion(iris_rf, method = "cv")
#' # Evaluation of performances using a separate test set
#' confusion(predict(iris_rf, newdata = iris_test), iris_test$Species)
#'
#' # Regression using random forest (from ?randomForest)
#' set.seed(131) # Useful for reproducibility (use a different number each time)
#' ozone_rf <- ml_rforest(data = airquality, Ozone ~ ., mtry = 3,
#'   importance = TRUE, na.action = na.omit)
#' summary(ozone_rf)
#' # Show "importance" of variables: higher value mean more important variables
#' round(randomForest::importance(ozone_rf), 2)
#' plot(na.omit(airquality)$Ozone, predict(ozone_rf))
#' abline(a = 0, b = 1)
#'
#' # Unsupervised classification using random forest (from ?randomForest)
#' set.seed(17)
#' iris_urf <- ml_rforest(train = iris[, -5]) # Use only quantitative data
#' summary(iris_urf)
#' randomForest::MDSplot(iris_urf, iris$Species)
#' plot(stats::hclust(stats::as.dist(1 - iris_urf$proximity),
#'   method = "average"), labels = iris$Species)
mlRforest <- function(train, ...)
  UseMethod("mlRforest")

#' @rdname mlRforest
#' @export
ml_rforest <- mlRforest

#' @rdname mlRforest
#' @export
#' @method mlRforest formula
mlRforest.formula <- function(formula, data, ntree = 500, mtry,
  replace = TRUE, classwt = NULL, ..., subset, na.action) {
  if (missing(mtry)) {
    mlearning(formula, data = data, method = "mlRforest", model.args =
        list(formula  = formula, data = substitute(data),
          subset = substitute(subset)), call = match.call(), ntree = ntree,
      replace = replace, classwt = classwt, ...,
      subset = subset, na.action = substitute(na.action))
  } else {
    mlearning(formula, data = data, method = "mlRforest", model.args =
        list(formula  = formula, data = substitute(data),
          subset = substitute(subset)), call = match.call(), ntree = ntree,
      mtry = mtry, replace = replace, classwt = classwt, ...,
      subset = subset, na.action = substitute(na.action))
  }
}

#' @rdname mlRforest
#' @export
#' @method mlRforest default
mlRforest.default <- function(train, response, ntree = 500, mtry,
  replace = TRUE, classwt = NULL, ...) {
  dots <- list(...)
  .args. <- dots$.args.
  dots$.args. <- NULL
  if (missing(response)) response <- NULL
  if (!length(.args.)) {
    if (!length(response)) {
      type <- "unsupervised"
    } else if (is.factor(response)) {
      type <- "classification"
    } else type <- "regression"
    .args. <- list(levels = levels(response),
      n = c(intial = NROW(train), final = NROW(train)),
      type = type, na.action = "na.pass",
      mlearning.call = match.call(), method = "mlRforest")
  }
  dots$ntree <- ntree
  dots$replace <- replace
  dots$classwt <- classwt

  # Return a mlearning object
  if (missing(mtry) || !length(mtry)) {
    res <- randomForest::randomForest(x = train,
      y = response, ntree = ntree, replace = replace,
      classwt = classwt, ...)
  } else {
    dots$mtry <- mtry
    res <- randomForest::randomForest(x = train,
      y = response, ntree = ntree, mtry = mtry, replace = replace,
      classwt = classwt, ...)
  }

  structure(res, formula = .args.$formula, train = train,
    response = response, levels = .args.$levels, n = .args.$n, args = dots,
    optim = .args.$optim, numeric.only = FALSE, type = .args.$type,
    pred.type = c(class = "response", membership = "prob", vote = "vote"),
    summary = NULL, na.action = .args.$na.action,
    mlearning.call = .args.$mlearning.call, method = .args.$method,
    algorithm = "random forest",
    class = c("mlRforest", "mlearning", "randomForest"))
}

#' @rdname mlRforest
#' @export
#' @method predict mlRforest
predict.mlRforest <- function(object, newdata,
type = c("class", "membership", "both", "vote"),
method = c("direct", "oob", "cv"), ...) {
  type <- as.character(type)[1]

  # If method == "cv", delegate to cvpredict()
  method <- as.character(method)[1]
  if (method == "cv") {
    if (!missing(newdata))
      stop("cannot handle new data with method = 'cv'")
    return(cvpredict(object = object, type = type, ...))
  } else if (method == "oob") { # Get out-of-bag prediction!
    if (!missing(newdata))
      stop("you cannot provide newdata with method = 'oob'")

    toProps <- function(x, ntree) {
      if (sum(x[1, ] > 1)) {
        res <- t(apply(x, 1, "/", ntree))
      } else {
        res <- x
      }
      class(res) <- "matrix"
      res
    }

    toVotes <- function(x, ntree) {
      if (sum(x[1, ] < ntree - 1)) {
        res <- round(t(apply(x, 1, "*", ntree)))
      } else {
        res <- x
      }
      class(res) <- "matrix"
      res
    }

    res <- switch(type,
      class = factor(as.character(object$predicted),
        levels = levels(object)),
      membership = .membership(toProps(object$votes, object$ntree),
        levels = levels(object)),
      both = list(class = factor(as.character(object$predicted),
        levels = levels(object)),
        membership = .membership(toProps(object$votes, object$ntree),
          levels = levels(object))),
      vote = .membership(toVotes(object$votes, object$ntree),
        levels = levels(object)),
      stop("unknown type, must be 'class', 'membership', 'both' or 'vote'"))

    attr(res, "method") <- list(name = "out-of-bag")
    res

  } else {
    predict.mlearning(object = object, newdata = newdata,
      type = type, norm.votes = FALSE, ...)
  }
}
