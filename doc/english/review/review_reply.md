The authors are grateful to the Editor and Reviewers, for their feedback! Here is the point-by-point response to the suggestions. The new version of the manuscript is  uploaded with highlighted corrections for better navigation.

## Editor
1. Please ensure the results are accurately reported, any overstated conclusions are rewritten and the limitations of the work, and the work’s limitations, fully explained.

We ensure that all paper results including mathematical and experimental are accurately reported. There are no overstated conclusions. The limitations of the work are now better explained. We have added a paragraph clarifying the boundaries and drawbacks of the decomposition algorithm in Section 5. It is also discussed at the end of Section 10. We analyzed the issue with the exploding forecast in Section 10.

## Reviewer 1

1. Clearly explain the importance of analyzing multivariate time series and how it provides advantages over univariate approaches.

Answer: We have added a designated paragraph explaining the importance of the multivariate time series in the Introduction. It includes several examples of complex systems described by multivariate time series such as brain activity and weather. We have added mathematical justification for why univariate methods may fail when applied to the multivariate time series.

2. Justify the necessity of decomposition in this context and how it enhances forecasting accuracy or interpretability.

Answer: We have added a designated paragraph in the Introduction justifying the necessity of the decomposition. Two advantages of the decomposition are given: understanding the structure of the time series and its denoising. The multidimensional Fourier transform and two multivariate denoising methods are mentioned as examples.

3. Explicitly highlight the key contributions of the paper to ensure the novelty and significance of the proposed approach are evident.

Answer: We have added a designated paragraph in the Introduction explicitly enumerating the key contributions of the paper. It includes mathematical and experimental results of the work.

4. Expand the problem statement by incorporating figures and practical examples for better clarity.

Answer: We have added the Lorenz attractor as a practical example supplemented with the figures. In the Problem statement section, it serves as a specific example of the unobserved dynamical system. Section Single-variate time series case section uses Takens’s theorem to show how to reconstruct the system.

5. Clarify whether the method can extract trend and seasonality components. Provide a detailed explanation of how the method handles these components and whether it effectively separates them in the decomposition process.

Answer: We have added a designated paragraph in Section The time series decomposition to clarify the limitations of the decomposition algorithm. The matrix factorization algorithms do not provide a particular decomposition since there is no unique one. It requires the choice of the factor matrices partitioning, which leads to many admissible decompositions. We refer to our analysis of the optimal decomposition problem in the next section and include other solutions with their drawbacks.

6. To effectively assess the generalization capability of the proposed method, evaluate it using benchmark multivariate time series datasets from diverse domains. Include forecasting plots to illustrate the method’s performance visually.

Answer: We have added the weather dataset to the computational experiment. It contains temperature, precipitation, and air pressure time series. We include the link the the dataset. Now the experiment comprises multivariate time series from the economy, mechanics, and weather forecasting domains. We have added the forecasting plots for all datasets.

## Additional corrections

Besides the mentioned corrections, we have decided to move technical figures to the Appendices section for better readability. It is the figures representing how the quality metrics of the forecast and the decomposition depend on the trajectory tensor rank. It includes figures showing that the quality of the computed CPD decomposition is increasing with the greater tensor ranks. Finally, we have made several small corrections to the text in Sections 1, 3, and 4 for better clarity. They are highlighted as well.

