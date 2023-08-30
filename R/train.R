#' Get the training variable for a mlearning object
#'
#' @description
#' The training variables (train) are the variables used to train a classifier,
#' excepted the prediction (class or dependent variable).
#'
#' @param object an object having a train attribute.
#' @param ... further parameter (depends on the method).
#'
#' @return A data frame containing the training variables of the model.
#' @export
#' @seealso [mlearning()], [response()], [confusion()]
#'
#' @examples
#' data("HouseVotes84", package = "mlbench")
#' house_rf <- ml_rforest(data = HouseVotes84, Class ~ .)
#' house_rf
#' train(house_rf)
train <- function(object, ...)
  UseMethod("train")

#' @rdname train
#' @export
#' @method train default
train.default <- function(object, ...)
  attr(object, "train")
