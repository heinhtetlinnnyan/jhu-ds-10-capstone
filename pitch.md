Next Word Predictor
========================================================
author: Hein Htet Linn Nyan
date: 25 September 2026
autosize: true

Type less. The app suggests your next word as you type.



The problem
========================================================

- Typing on a phone is slow.
- Many words are predictable from the words just before them.
- A good suggestion saves a tap on almost every word.

**The product:** a web app that shows the three most likely next words as you type, and adds one with a single click.

How the model works
========================================================

- Trained on 93,046 words from blogs, news and Twitter.
- Counts which words follow each 1-, 2- and 3-word phrase.
- To predict, it looks up the last three words typed.
- If that phrase was never seen, it drops the oldest word and tries again ("backoff").
- Improvements: equal text from each source, rare phrases pruned, and the backoff weight tuned on separate data.

How well it works
========================================================

Tested on text the model never saw:

| Measure | Result |
|:--|:--|
| Right word is the first suggestion | 16% |
| Right word is in the top three | 26% |
| Time per prediction | 9.4 ms |
| Model file size | 2.6 MB |

For comparison, the basic model got 24% in the top three on the same test.

Try the app
========================================================

**Link:** https://heinhtet.shinyapps.io/shiny_app/

1. Type a phrase, e.g. *"thanks for the"*.
2. Three suggested words appear as buttons.
3. Click one to add it to your text, and keep going.
4. The bar chart shows how confident the model is in each word.

Fast, small and simple: ready to build into a keyboard.
