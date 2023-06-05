
## for sequence,

"[history] i love eating hot dog in the ___ "

if the token "love" in this sequence was also predicted by the model, then "[history] i ____" -> "love" must be how model was trained, therefore, "in the distribution".
However, if we replace the word "love" with other tokens e.g. "<A>". Then "[history] i love" may not be how the model was trained (out of distribution).
In this case, if we let the model predict "[history] i <A> eating hot dog in the ___ ", it is possible that the model is working entirely wrong (not reflecting usual mechanism), not due to "love" is essential, but "<A>" ruin the sequence, where the model have no idea of how to process this unfamiliar sequence. 



## For example:

"[history] i love eating hot dog in the ___ "

modified to:
"[history] i Coca eating hot dog in the ___ "

Model: "this sequence looks unlike what I have been trained on, but when I see "Coca", I usually need to predict "Cola" if it is not present, so I'll go for it."

Note: the mechanism could be different than what model works for the original sequence.




## Also, an example of the greedy picking one (Keyon etc.):

"[history] i love eating hot dog in the ___ "

we pick ["the"] as the starting point of the rationals

- then we evaluate:
  - [ "in", "the" ]
  - [ "dog", "the" ]
  - [ "hot", "the" ]

- the model:
  - [ "in", "the" ] looks like part of a continual sequence, I know kow to work with it, just predict the next token as usual.
  - [ "dog", "the" ] does not look like a continual sequence. I have not been trained on this kind of thing. I may produce some random stuff.
  - [ "hot", "the" ] same, not familiar, my output might be out of distribution.

Note: in the [ "dog", "the" ] and [ "hot", "the" ] case, model is not trained on them and we don't know if the usual mechanism will be reflected in both cases.
if they are not working as usual, how can we relies on those predicted likelihood to rank/extract the rationals?

Note 2: they (Keyon etc.) fine-tune the model to make those cases (un-continual sequence) usual for model, then the model will (hopefully) work as usual for those cases.









# 06-05

- replacement
  - random replace
  - replace with predicted words
    - is the final likelihood valid in this case?
- update inportance score
  - base on replaced sequence likelihood
