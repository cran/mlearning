#' Get the response variable for a mlearning object
#'
#' @description
#' The response is either the class to be predicted for a classification problem
#' (and it is a factor), or the dependent variable in a regression model (and
#' it is numeric in that case). For unsupervised classification, response is not
#' provided and should return `NULL`.
#'
#' @param object an object having a response variable.
#' @param ... further parameter (depends on the method).
#'
#' @return The response variable of the training set, or `NULL` for unsupervised
#'   classification.
#' @export
#' @seealso [mlearning()], [train()], [confusion()]
#'
#' @examples
#' data("HouseVotes84", package = "mlbench")
#' house_rf <- ml_rforest(data = HouseVotes84, Class ~ .)
#' house_rf
#' response(house_rf)
response <- function(object, ...) {
  UseMethod("response")
}

#' @rdname response
#' @export
#' @method response default
response.default <- function(object, ...) {
  attr(object, "response")
}
