# Methodology idea
1. build a data set for arsenic contamination by symulating arsenic contamination events with EPyT-Flow

2. Choose and use a chlorine injection pattern 
    --> we choosed a constant injection pattern, here you can find a tutorial that explains you how to set a constant injection pattern [🔗 tutorial](https://epyt-flow.readthedocs.io/en/stable/examples/chlorine_injection.html)

3. Choose a model for chlorine decay:
    EPANET internally uses by default bulk decay:
    ![alt text](<Screenshot 2026-05-29 at 16.03.27.png>)
    [@Monteiro_Figueiredo_Dias_Freitas_Covas_Menaia_Coelho_2014](1-s2.0-S1877705814001349-main.pdf) mentioned the problem that MSX is not super user friendly if you don't know msdos, or if you aren't a programmer... they also used EPANET to simulate chlorine decay

    From the [@epanet_manual](EPANETMSX.pdf) at the 48th page of the pdf we found out how to set up bulk decay in our project. (still an investigation of how to set the parameters for a correct bulk decay needs to be done, but it looks like it's kinda implemented out of the box)
4. Choose an event/anomaly detection method t identify water contamination
   First intuition (Without having read anything): Just compare the normal level of chlorine to the current level of chlorine
   and look whether it is different. Since chlorine injection is constant, only an exterior influence (like arsenic) can cause a
   lower chlorine level. Did I get the point? I'm unsure

   Update: Zhao et al. use that strategy too
   <img width="1192" height="409" alt="grafik" src="https://github.com/user-attachments/assets/c5252783-7311-472d-8c25-b0506d6f5256" />

5. Performance test: evaluation of the chosen method
   First intuition (Without having read anything): We talked about accuracy and event detection latency. Since both of these
   will be cardinal/metric data, good metrics would be the mean and the standard deviation. For comparing two models.
   Especially the latency time might be linked to the runtime of the model, so maybe check that too?
   To compare different evaluation metrics of different sensor locations or different implementations, statistical tests like
   the t-test can be used. The hypothesis here would be, placement/implementation a is better than placement/implementation b.
   As p-value, something like 0.05 could be chosen. 
7. Optimization: optimize size and location where these contamination events can be detected better.
   Zhao et al. sum up that there are different methods for finding the best sensor placements like mixed-integer programming or
   GRASP. These can be implemented by programming.
