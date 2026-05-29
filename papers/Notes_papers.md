# Notes on papers

Sensing technologies paper:

Maybe easily implementable: Chemical sensors, p.2816. Add another chemical to provoke a reaction

https://de.wikipedia.org/wiki/Arsen(III)-chlorid

https://de.wikipedia.org/wiki/Chlorung

# Papers on water event detection

## Event Detection in Water Distribution Systems from Multivariate Water Quality Time Series:

https://pubs.acs.org/doi/full/10.1021/es3014024
--> using a neural network

## Network hydraulics inclusion in water quality event detection using multiple sensor stations data

Author links open overlay panel
https://www.sciencedirect.com/science/article/pii/S0043135415002730
--> using sensor data like chlorine sensor data and flow data

## Advanced Sensor for Arsenic and Fluoride Detection 

https://link.springer.com/chapter/10.1007/978-981-19-9151-6_48
- just the abstract is useful
- issues with arsenic contamination, why developing an arsenic sensor is not easy (it's a work in progress)
Conclusion: our project is a software solution that can take advantage of already available sensors, for flow and chlorine without the need of buying, building and installing more expensive sensors.

## Water Quality Event Detection in Drinking Water Network
`@Zhao_Hou_Huang_Zhang_2014`
- https://link.springer.com/article/10.1007/s11270-014-2183-7
- [PDF](s11270-014-2183-7.pdf)
- this article is a review, that explains water event detection (definitions, methods ect.)
- useful as first resource to understan better how to tackle anomaly/event detection

## Other papers that could become handy in the future:
1. [https://www.sciencedirect.com/science/article/pii/S0301479709000103](https://www.sciencedirect.com/science/article/pii/S0301479709000103)
2. [https://www.sciencedirect.com/science/article/pii/S1877705815026739](https://www.sciencedirect.com/science/article/pii/S1877705815026739)
3. [https://www.sciencedirect.com/science/article/pii/S0043135413000341](https://www.sciencedirect.com/science/article/pii/S0043135413000341)
4. [https://www.sciencedirect.com/science/article/pii/S1877705814023443](https://www.sciencedirect.com/science/article/pii/S1877705814023443)
5. [https://www.mdpi.com/1424-8220/20/5/1342](https://www.mdpi.com/1424-8220/20/5/1342)
6. [https://www.spiedigitallibrary.org/conference-proceedings-of-spie/6203/62030K/Adaptive-monitoring-to-enhance-water-sensor-capabilities-for-chemical-and/10.1117/12.665358.short](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/6203/62030K/Adaptive-monitoring-to-enhance-water-sensor-capabilities-for-chemical-and/10.1117/12.665358.short)
7. [https://iwaponline.com/aqua/article/74/7/451/108698/Multi-parameter-multi-sensor-data-fusion-for](https://iwaponline.com/aqua/article/74/7/451/108698/Multi-parameter-multi-sensor-data-fusion-for)

# Notes for the report:
- a better sensor placement can help detect contamination quicly and robustly, reducing the damage caused by contamination `@Zhao_Hou_Huang_Zhang_2014`

# Methodology idea
1. build a data set for arsenic contamination by symulating arsenic contamination events with EPyT-Flow
2. Choose and use a chlorine injection pattern (also constant is allowed)
3. Choose a model for chlorine decay:
    EPANET internally uses by default bulk decay:
    ![alt text](<Screenshot 2026-05-29 at 16.03.27.png>)
    [@Monteiro_Figueiredo_Dias_Freitas_Covas_Menaia_Coelho_2014](papers/1-s2.0-S1877705814001349-main.pdf) mentioned the problem that MSX is not super user friendly if you don't know msdos, or if you aren't a programmer... they also used EPANET to simulate chlorine decay

    From the [@epanet_manual](papers/EPANETMSX.pdf#page=48) we found out how to set up bulk decay in our project.

4. Choose an event/anomaly detection method t identify water contamination
5. Performance test: evaluation of the chosen method
6. Optimization: optimize size and location where these contamination events can be detected better.
