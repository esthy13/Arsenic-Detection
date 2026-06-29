1. What's the mathematical model and graphical representation of the ANN that was implemented?

   Stack of a few of these layer depending on what was the best configuration according to tests:
   Linear Layer
   GeLu
   Dropout

   Head of the network:
   Linear Layer
   Sequential Layer
2. How many Layers does the ANN have?
   TODO
   not yet decided depends on the test from the future tasks
3. Do I need to do Fuse Decision like in the original paper? Am I already doing it?
   I actually need to predict only chlorine, which means that there is no Fusion Decision in my implementation
4. Am I running ANN two times? One for chlorine and one for flow? Or just one run?
   Just one run, because there is no decision fusion
5. Does the implementation of the Genetic Algorithm corresponds with what is described in the paper? (explanation with code snippets for the report) TODO
6. Confusion matrix to understand whether the event detection results are good or bad: DONE

TODO Future tasks:

- Rock curve, sensitivity and specificity to evaluate and fine-tune the ANN
- Other methods to effectively tune ANN (maybe ask the professor)
- IDEA: check if flow is influenced by Arsenic Contamination or not
