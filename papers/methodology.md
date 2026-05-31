# Methodology idea
1. Choose and use a chlorine injection pattern 
    --> we choosed a constant injection pattern, here you can find a tutorial that explains you how to set a constant injection pattern [🔗 tutorial](https://epyt-flow.readthedocs.io/en/stable/examples/chlorine_injection.html)

2. build a data set for arsenic contamination by symulating arsenic contamination events with EPyT-Flow

3. Choose a model for chlorine decay:
    EPANET internally uses by default bulk decay:
    ![alt text](<Screenshot 2026-05-29 at 16.03.27.png>)
    [@Monteiro_Figueiredo_Dias_Freitas_Covas_Menaia_Coelho_2014](1-s2.0-S1877705814001349-main.pdf) mentioned the problem that MSX is not super user friendly if you don't know msdos, or if you aren't a programmer... they also used EPANET to simulate chlorine decay

    From the [@epanet_manual](EPANETMSX.pdf) at the 48th page of the pdf we found out how to set up bulk decay in our project. (still an investigation of how to set the parameters for a correct bulk decay needs to be done, but it looks like it's kinda implemented out of the box)

4. Choose an event/anomaly detection method t identify water contamination
    We have two proposition:
    - A simple method: **linear prediction**, for a reference -> [Detecting Changes in Water Quality Data](Journal%20AWWA%20-%202008%20-%20McKenna%20-%20Detecting%20Changes%20in%20Water%20Quality%20Data.pdf)
        the water quality parameter is predicted for the next time step, and he predicted value is compared to the measured value. The residual is used as anomaly score:
        $$a(t)=r(t)=\hat{x}(t)-x(t)$$
    - A more complex (and supposedly better performing method): based on **artificial neural networks (ANN)**, with dynamic treshold, for a reference, check [A dynamic thresholds scheme for contaminant event detection in water distribution systems](papers/1-s2.0-S0043135413000341-main.pdf), where the anomaly score is just the water quality data itself, but the threshold is adaptive according to the current water quality: $$a(t) = x(t)$$

    Meaning of math variables:
    $a(t)$ --> anomaly score at time $t$
    $x(t)$ --> water quality data at time $t$
    $\hat{x}(t)$ --> predicted water quality data at time $t$
    $r(t)$ --> the residual at time $t$

5. Performance test: evaluation of the chosen method
    Usually done with:
    - **accuracy** --> it's important that the number of false alarms is reduced (as well as the number of missed alarms)
        - **Probability of detection**: $PD = \frac{TP}{TP + FN}$
        - **false alarm rate**: $ FAR = \frac{FP}{FP + TN}$
    - **detection latency time** --> the system should be as real time as possible, to notify people to not drink water (as it would be dangerous), and to notify the water company that they have to works on their systems in order to stop the Arsenic contamination. Latency time is suggested to be counted from the time when the peak of the event has taken place (@raciti2012anomaly)

6. Optimization: optimize size and location where these contamination events can be detected better.
    1. **Mixed-Integer Programming (MIP)** --> Formulates sensor placement as a mathematical optimization problem with binary variables and constraints (Requires advanced optimization solvers (CPLEX, Gurobi), Need to model network constraints mathematically, Long computation times even for medium networks) (@berry2005sensor)
    2. A better alternative that **I** propose to MIP is **minizinc** which should be easier to implement, however still is king of time consuming, and we still need somekind of matematical modeling of the network (if you want to checkout how minizinc works https://www.minizinc.org/), it can be easily embedded in python.
    3. An even simpler approach should be **Greedy Algorithm / Greedy-based Methods** --> Iteratively places one sensor at a time, choosing the location that provides maximum impact reduction in each step. Two variants:
        - Greedy placement: Start with empty set, add one sensor each step at best location
        - Greedy replacement: Start with random placement, swap one sensor at a time for better location
    - Fast compared to MIP
    - Good enough solutions (often 90%+ of optimal)
    - Suitable for large networks
    Cons:
    - May get stuck in local optima
    - Not guaranteed globally optimal
    (refences: @krause2008efficient, @eliades2008iterative, @dorini2010slots)


# Why does Arsenic contamination happen and why it is dangerous?
<!-- TODO -->

### Extra references
```bibtex
@incollection{raciti2012anomaly,
  title={Anomaly detection in water management systems},
  author={Raciti, Massimiliano and Cucurull, Jordi and Nadjm-Tehrani, Simin},
  booktitle={Critical Infrastructure Protection: Information Infrastructure Models, Analysis, and Defense},
  pages={98--119},
  year={2012},
  publisher={Springer}
}

@article{berry2005sensor,
  title={Sensor placement in municipal water networks},
  author={Berry, Jonathan W and Fleischer, Lisa and Hart, William E and Phillips, Cynthia A and Watson, Jean-Paul},
  journal={Journal of Water Resources Planning and Management},
  volume={131},
  number={3},
  pages={237--243},
  year={2005},
  publisher={American Society of Civil Engineers}
}

@article{krause2008efficient,
  title={Efficient sensor placement optimization for securing large water distribution networks},
  author={Krause, Andreas and Leskovec, Jure and Guestrin, Carlos and VanBriesen, Jeanne and Faloutsos, Christos},
  journal={Journal of Water Resources Planning and Management},
  volume={134},
  number={6},
  pages={516--526},
  year={2008},
  publisher={American Society of Civil Engineers}
}

@inproceedings{eliades2008iterative,
  title={Iterative deepening of Pareto solutions in water sensor networks},
  author={Eliades, Demetrios and Polycarpou, Marios},
  booktitle={Water distribution systems analysis symposium 2006},
  pages={1--19},
  year={2008}
}

@article{dorini2010slots,
  title={SLOTS: Effective algorithm for sensor placement in water distribution systems},
  author={Dorini, Gianluca and Jonkergouw, Philip and Kapelan, Zoran and Savic, Dragan},
  journal={Journal of Water Resources Planning and Management},
  volume={136},
  number={6},
  pages={620--628},
  year={2010},
  publisher={American Society of Civil Engineers}
}
```
---
   First intuition (Without having read anything): Just compare the normal level of chlorine to the current level of chlorine
   and look whether it is different. Since chlorine injection is constant, only an exterior influence (like arsenic) can cause a lower chlorine level. Did I get the point? I'm unsure

   Update: Zhao et al. use that strategy too
   <img width="1192" height="409" alt="grafik" src="https://github.com/user-attachments/assets/c5252783-7311-472d-8c25-b0506d6f5256" />

5. Performance test: evaluation of the chosen method
   First intuition (Without having read anything): We talked about accuracy and event detection latency. Since both of these
   will be cardinal/metric data, good metrics would be the mean and the standard deviation. For comparing two models.
   Especially the latency time might be linked to the runtime of the model, so maybe check that too?
   To compare different evaluation metrics of different sensor locations or different implementations, statistical tests like
   the t-test can be used. The hypothesis here would be, placement/implementation a is better than placement/implementation b.
   As p-value, something like 0.05 could be chosen. 
6. Optimization: optimize size and location where these contamination events can be detected better.
   Zhao et al. sum up that there are different methods for finding the best sensor placements like mixed-integer programming or
   GRASP. These can be implemented by programming.
