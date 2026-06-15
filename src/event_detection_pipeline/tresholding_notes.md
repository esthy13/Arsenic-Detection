Explanation of tresholding mechanisms

## 1. `rolling_thresholds` (Dynamic Baseline)

This function calculates a moving upper and lower threshold for every data point using a rolling window.

* The Logic: For each step, it looks back at a specific segment of past data (defined by `window_fraction`). It calculates the mean and standard deviation of that local window to establish what "normal" looks like  *right now* .
* Outlier Filtering: Before calculating the mean and standard deviation, it strips out extreme values using `outlier_upper` and `outlier_lower`. This prevents past anomalies from inflating the standard deviation and "masking" new anomalies.
* The Thresholds:
  $$
  \text{Upper} = \text{Local Mean} + (\text{Multiplier} \times \text{Local StdDev})$$ 
  $$\text{Lower} = \text{Local Mean} - (\text{Multiplier} \times \text{Local StdDev})
  $$
* The Break Condition: If a rolling window contains entirely extreme outliers (leaving `filtered` empty), the function completely resets the thresholds to zero and stops.

---

## 2. `bayesian_event_probability` (Belief Tracking)

Once a residual crosses a dynamic threshold, this function decides *how confident* the system is that a real event is happening. It treats anomaly detection as a recursive Bayesian estimation problem.

* Bayes' Theorem updates:

  * If a data point exceeds the threshold, the function calculates the probability that an event is happening given a positive test result ($P(\text{Event} \mid \text{Anomaly})$).
  * If it is within normal bounds, it calculates the probability that an event is happening given a negative test result ($P(\text{Event} \mid \text{Normal})$).
* Smoothing/Memory: It applies an Exponential Moving Average (`smoothing`) to blend the newly calculated probability with the `previous_probability`. This prevents a single noisy data point from spiking or dropping the alert state instantly.
* Clamping: The probability is strictly capped at a maximum of `0.95` and a minimum of your `initial_probability` to keep the filter agile and responsive to sudden state changes.

---

## 3. `evaluate_threshold` (Performance Metrics)

This function tests a specific configuration of multipliers and window sizes against your data to see how well it performs.

* Confusion Matrix: It classifies points into True Positives (correctly flagged events), False Positives (false alarms), False Negatives (missed events), and True Negatives (correctly ignored normal data).
* Metrics: It calculates Sensitivity (True Positive Rate/Recall) and Specificity (True Negative Rate).
* The Objective Function: It computes a score to evaluate performance: `-(sensitivity + specificity) + penalty`. Because the optimization loop tries to minimize this value, maximizing sensitivity and specificity forces the objective score downward.
* The Edge-Case Penalty: If a configuration is so loose or tight that *every single point* is classified exactly the same way (either all anomalies or all normal), it adds a `1.0` penalty to disqualify that useless configuration.

---

## 4. `fit_threshold` (Hyperparameter Tuning)

This is a grid search optimizer that finds the ideal parameter settings for your specific dataset.

* The Search Space: It loops through predefined combinations of window sizes (`candidate_windows`), threshold tighteners (`candidate_multipliers`), and outlier filters (`candidate_filters`).
* Selection: It runs `evaluate_threshold` for every possible combination. The configuration that yields the lowest objective score (highest combined sensitivity and specificity) is selected as the winner and returned alongside its calculated thresholds.

---

## Summary of the Flow

1. `fit_threshold` runs a grid search over hyperparameters.
2. It calls `evaluate_threshold` for each candidate setup.
3. `evaluate_threshold` calls `rolling_thresholds` to generate an adaptive boundary envelope.
4. The combination that best balances catching real events while minimizing false alarms is returned.
5. The optimized thresholds can then be fed into `bayesian_event_probability` to output a clean, smoothed real-time probability curve of whether an active threat or failure event is ongoing.