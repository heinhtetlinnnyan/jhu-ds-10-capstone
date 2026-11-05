# Next Word Predictor

A Shiny app that suggests the next word as you type.

## Use it

1. Type a phrase in the box.
2. The three most likely next words appear as buttons. Click one to add it to your text.
3. The bar chart shows how confident the model is in each word.

## Run it yourself

The folder needs three files:

- `app.R`: the app
- `ngram_engine.R`: the prediction functions
- `final_model.rds`: the model (made by `improve_model.R`)

In R:

```r
install.packages(c("shiny", "data.table", "stringi"))
shiny::runApp("shiny_app")
```

## Put it online (shinyapps.io)

```r
install.packages("rsconnect")
rsconnect::setAccountInfo(name = "<account>", token = "<token>", secret = "<secret>")
rsconnect::deployApp("shiny_app")
```

Get the name, token and secret from your shinyapps.io account under **Account > Tokens**.

## Use the model without the app

```r
source("ngram_engine.R")
model <- readRDS("final_model.rds")
predict_next(model, "thanks for the", k = 3, alpha = model$alpha)
```

This returns three words, with their confidence in `attr(result, "p")`.

## Model size

The free shinyapps.io plan has 1 GB of memory. If the app runs out of memory, rebuild the model with a higher minimum count in `improve_model.R` (e.g. 4 instead of 3). This makes the model smaller, at a small cost in accuracy.
