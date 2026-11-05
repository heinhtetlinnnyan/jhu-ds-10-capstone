# app.R — Next-word prediction app
# Needs ngram_engine.R and final_model.rds in the same folder.

library(shiny)
source("ngram_engine.R")

model <- readRDS("final_model.rds")

ui <- fluidPage(
        titlePanel("Next Word Predictor"),
        tabsetPanel(
                tabPanel("Predict",
                         br(),
                         textInput("phrase", "Type a phrase:", value = "", width = "100%",
                                   placeholder = "e.g. thanks for the"),
                         p("Suggested next words (click one to add it):"),
                         actionButton("pick1", "", class = "btn-primary"),
                         actionButton("pick2", ""),
                         actionButton("pick3", ""),
                         br(), br(),
                         plotOutput("conf_plot", height = "220px"),
                         textOutput("note")
                ),
                tabPanel("How it works",
                         br(),
                         p("The app looks at the last three words you typed and finds which words",
                           "most often came next in a large collection of blogs, news and tweets."),
                         p("If those three words were never seen together, it tries the last two words,",
                           "then the last one, then falls back to the most common words overall."),
                         p("The bars show how often each word followed your phrase in the training text.",
                           "Higher bars mean the model is more confident."),
                         if (!is.null(model$top3))
                                 p(sprintf("Tested on text it never saw, the right word was among the three suggestions %.0f%% of the time.",
                                           100 * model$top3))
                )
        )
)

server <- function(input, output, session) {
        
        # Wait until typing pauses before predicting
        phrase <- debounce(reactive(input$phrase), 300)
        
        preds <- reactive({
                w <- predict_next(model, phrase(), k = 3, alpha = model$alpha)
                data.frame(word = as.character(w), p = attr(w, "p"))
        })
        
        # Show the three words on the buttons
        observe({
                d <- preds()
                for (i in 1:3) {
                        label <- if (i <= nrow(d)) d$word[i] else ""
                        updateActionButton(session, paste0("pick", i), label = label)
                }
        })
        
        # Clicking a button adds that word to the text
        lapply(1:3, function(i) {
                observeEvent(input[[paste0("pick", i)]], {
                        d <- preds()
                        if (i > nrow(d)) return()
                        new_text <- trimws(paste(input$phrase, d$word[i]))
                        updateTextInput(session, "phrase", value = paste0(new_text, " "))
                })
        })
        
        output$conf_plot <- renderPlot({
                d <- preds()
                par(mar = c(4, 7, 1, 1))
                barplot(rev(100 * d$p), names.arg = rev(d$word), horiz = TRUE, las = 1,
                        col = "steelblue", border = NA, xlim = c(0, 100),
                        xlab = "Confidence (%)")
        })
        
        output$note <- renderText({
                top <- preds()$p[1]
                if (top >= 0.5) "The model is fairly sure about the first suggestion."
                else if (top >= 0.2) "Several words are plausible here."
                else "Low confidence: many different words could come next."
        })
}

shinyApp(ui, server)
