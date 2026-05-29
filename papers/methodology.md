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
5. Performance test: evaluation of the chosen method
6. Optimization: optimize size and location where these contamination events can be detected better.