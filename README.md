# TEEP_XGBoostCatalystOptimizer
XGBoost-Based optimizer for catalyst compositions developed during TEEP exchange internship.

Instructions:

For this software, the target and training variables must be named in config and the run code. Additionally, the order of interaction terms must be adjusted to match the number of components there are to interact.

An update is coming to make these adjustments easier, as well as include the hyperparameter optimzing loop.

A venv must be created, with config and ML_simplePipeline in said venv, and ML_paper_data_HE available in the same folder as the venv. The code can then be run to demonstrate how the current method works.

Literature Behind Work:

“Machine Learning” was first coined as a term by Arthur Samuel in 1959, who created a checkers-playing algorithm that learned experientially. This idea of machines learning by experience is the core behind machine learning, as described by Zhou: “develop[ing] learning algorithms that build models from data.” [1] Machine learning as a field grew out of the artificial intelligence studies of the 1950s to 1970s. The main pursuit at the time was developing a machine that could use logical reasoning in the hopes that this would produce Artificial Intelligence. Notably, in his “Turing Test” paper, Turing recognized the necessity of experiential learning for this goal to be realized. In the 1970s, Feigenbaum led the rest of the field to begin to recognize this, and research shifted towards learning methods. In the 1980s, learning methods grew, became more characterized, and were explored in depth, and the favorite, learning by example, emerged. The main method in learning by example is symbolism, using decision trees and logic to build a model. Symbolism importantly continues the lineage of logic and knowledge methods from the history of artificial intelligence development, which is reflected in its continued presence in machine learning. More recently emerging approaches include statistical learning and a new method of connectionism, deep learning. However, neither applies to the XGBoost approach being taken.

XGBoost operates on the same principles as many other machine learning programs. The main mathematical concept at play is “Gradient-Boosted Trees,” or a series of decision trees generated to help divide a population of data points into subcategories that can then be used as predictors for new points. This is common ground with a more traditional “Random Forest” algorithm (a forest in this case being comprised of many decision trees). However, the Gradient-Boosted Trees are constructed sequentially (vertically as opposed to Random Forest’s lateral growth), allowing each optimization step to act as a correction for the previous. The core function driving the algorithm is the “Objective Function,” or given generally in the XGBoost documentation as:

obj(θ)=L(θ)+ Ω(θ)

Where L is the “Loss Term,” or in this program’s case, the mean squared error.

L(θ)= ∑_i (y_i-ŷ_i)^2

This could also be replaced with logistic loss, depending on the needs of the model. The omega term is called the “Regularization Term” and aims at reducing the overfitting of the model. At a skimming level, the regularization term is controlled by a coefficient, lambda. Lambda tells the algorithm how rapidly to force the weight of each subsequent tree towards zero. With the two parts of the objective function working together, a full tree structure is built that then can be applied to new data points to predict an output value [2].
A model is like an “explanation” that the algorithm gives, a curve fit to the data points. The general form of a linear model is: 

f(x) = w^Tx + b

In linear algebra terms, f(x) is the target variable, some value that is a function of some x (either a scalar or vector), and the model is attempting to construct a vector or matrix, w, and some scalar or vector, b, that performs a linear transformation that best maps each x value to its corresponding f(x). This is the core of a model: creating a linear transform that successfully maps the input space to the output space.

However, almost every problem will not have a readily available exact solution. This is easily apparent when considering large datasets or data of higher dimensions and the limitations of a linear map. This is where the actual optimization step and machine learning take place, with each iteration attempting to better approximate the exact solution. If the input space, D, is:

D = {(x_1, y_1), …, (x_m, y_m)}

Then the model learns: 

f(x_i) ≅ y_i for i = 1, …, m

And minimizes the difference between f(x) and y using least squares:

Single-Dimensional:
(w*, b*) = arg min Σ[f(x_i) - y_i]^2
		 = arg min Σ[y_i – wx_i - b]^2

Multi-Dimensional:
 X = [(x_1^T, 1); (x_2^T, 1); …; (x_m^T, 1)]
w* = arg min (y – Xw)^T(y – Xw)

Where w* and b* indicate solutions for w and b. 

To better understand the XGBoost software, it is necessary to understand decision trees, regression modeling, greedy algorithms, and discuss the software’s use of all these methods. After understanding regression modelling, the next step is to discuss decision trees. 

Tree-based logic is a method of machine learning that involves the categorization of samples based on their features. Essentially, data points have some characteristics that are quantified by the scalar or logical values assigned to them, such as temperature and flow rate, or shape and color. Each of these attributes on its own is a feature of the dataset and can be used in training. A tree is a branching logic decision that splits the data based on features. A simpler algorithm using trees is Random Forest. In this method, a random subset of the data is generated, and a decision tree for that set is created as a predictor from that subset. All of the trees are then used as a “forest” of horizontally grown trees for prediction. In XGBoost, trees are also used, but instead of horizontally grown trees, they are grown sequentially (or vertically), called “Boosting”. Each tree follows up on the previous step, reducing error.

One more component that is necessary before the mathematical discussion is the term “Greedy”. In machine learning and algorithmic thinking, greedy means that the optimum is approached by simply taking the next step in the direction of greatest error reduction. The rest of the function remains the same, with no changing of previous decisions. Rather, each step gets incrementally closer to the global optimum. An important method of this approach for XGB is “Gradient Descent,” whose use is indicated in the name “Extreme Gradient Boost”. This is typically computed by a “steepest descent” method, where the gradient is computed as the derivative of the expected loss with respect to each step’s previously defined variable. While not particularly clear here, this will hopefully make more sense in the context of mathematical representation.
Before describing the actual training and optimization steps, some useful terms are:

   Φ(F) = Ey, x L(y, F(x)) = Ex[Ey(L(y, F(x))) | x]
Φ(F(x)) = Ey[L(y,F(x)) | x]

These two expressions are roughly equivalent, with the second readily obtained from the first. Φ() indicates the “expected loss” function, as seen in the use of the loss function alongside the expectation functions for both x and y. In gradient boosting, the problems are solved in “function space,” or solved via the construction of a function rather than the optimization of parameters, as in a “parameter space” problem. The solution in function space can be expressed as:

F*(x) = Σ f_m(x)

Where F*(x) is the solution, and each function f_m(x) is a boosting step. Thinking of each consecutive f_m(x) as adding a new boost to f_m-1(x), the following equation can be obtained:

   F_m(x) = F_m-1(x) + 𝛽_m*h(x;a_m)
(𝛽_m, a_m) = arg min 𝛴 L(y_i, F_m-1(x_i) + 𝛽_m * h(x_i; a))

To make sense of these two equations, h(x; a) should be considered as a tree, more specifically, the new boosting step’s tree. This gives a direction for the greedy boost to step in, and the amount in said direction is quantified by the scalar 𝛽_m, unique for each boosting step.

The continual minimization of the loss function by greedy steps towards an improved solution is the core of Gradient Boosted Trees. To build a Gradient Boosted Trees model, the structure follows this pattern:

	F0(x) = arg min⍴ 𝛴 L(y_i, ⍴)
	For m = 1 to M, do:
	ỹ_i = -[𝜕L(y_i, F(x_i))/𝜕F(xi)] ; F(x) = F_m-1(x), i = 1, N
        The derivative of loss with respect to each variable/feature. This is the steepest descent step.
	a_m = arg min(a, 𝛽) 𝛴 [ỹ_i - 𝛽h(x_i;a)]^2
	⍴_m  = arg min⍴  L(y_i, F_m-1(x_i) + ⍴h(x_i;a_m))
	F_m(x) = F_m-1(x) + ⍴_m*𝛴b_jm * 1(x ∈ R jm)
	End For
    End Algorithm

If the problem posed by catalyst optimization were classification (discrete outcomes), the equation Fm(x) = F_m-1(x) + ⍴_m * h(x; a_m) would be used in step 6 instead. This equation is more readily understood in that context, more clearly being the added stage in boosting than the equation used at line 6. However, line 6 functions the same as this classification approach, taking the average decrease in error at the new step.

This structure in steps 1-7 is the core of what XGBoost does: take an initial guess, then continually loop, adding better estimators until the error on the training set is within some target set by the program. The developers of XGBoost attribute this method to Greedy Function Approximation: A Gradient Boosting Machine by Friedman, which is the central work consulted here [3].

Citations:
[1] Z.-H. Zhou, Machine Learning. S.L.: Springer Verlag, Singapor, 2022.
[2] xgboost developers, “Introduction to boosted trees — xgboost 1.5.1 documentation,” xgboost.readthedocs.io, 2022. https://xgboost.readthedocs.io/en/stable/tutorials/model.html (accessed Aug. 26, 2026).
[3] J. H. Friedman, “Greedy Function Approximation: A Gradient Boosting Machine,” The Annals of Statistics, vol. 29, no. 5, pp. 1189–1232, 2001, Accessed: Aug. 26, 2026. [Online]. Available: https://www.jstor.org/stable/2699986

