# Gaussian Mixtures for Generative Adversarial Networks (GANs) in Unsupervised Learning Scenarios

This group project was completed for the Data Science Lab course as part of the IASD Master Program (AI Systems and Data Science) at PSL Research University during Semester 1 of the 2023-24 academic year.
The project accomplished the following:
- Enhanced MNIST dataset's digit generation using Gaussian Mixture Generative Adversarial Networks (GM-GAN) in unsupervised learning scenarios, building upon a well-optimized vanilla GAN.
- Established a robust baseline with the vanilla GAN through rigorous hyperparameter tuning, which significantly informed the approach to refining a static GM-GAN.
- Generated a higher frequency of better-looking digits with a higher variety using the final Gaussian Mixture model and recommended hyperparameters from the tuning analysis.

The full report associated with this project can be found in the *report.pdf* file of this repository, which details our approach and methods used to complete this project, as well as the analyzed results.

## General Information
If you want to copy and recreate this project, or test it for yourself, some important information to know.

### train.py
To run and train the model, use the the file *train.py* to view the results we analyzed in our report.
  > python3 train.py

### generate.py
Use the file *generate.py* to generate 10000 samples of MNIST in the folder samples. 
Example:
  > python3 generate.py --batch_size 64


### requirements.txt
Among the good pratice of datascience, we encourage you to use conda or virtualenv to create python environment. 
  > pip install -r requirements.txt


## Acknowledgements
This project was made possible with the guidance and support of the following :
 
- **Prof. Benjamin Negrevergne**
  - Associate professor at the Lamsade laboratory from PSL – Paris Dauphine university, in Paris (France)
  - Active member of the *MILES* project
  - Co-director of the IASD Master Program (AI Systems and Data Science) with Olivier Cappé.

- **Alexandre Verine**
  - PhD candidate at LAMSADE, a joint research lab of Université Paris-Dauphine and Université PSL, specializing in Machine Learning.

This project was a group project and was accomplished through team work involving the following students :

- **Matteo Sammut**
  - Masters student in the IASD Master Program (AI Systems and Data Science) at PSL Research University.

- **Alexandre Ngau**
  - Masters student in the IASD Master Program (AI Systems and Data Science) at PSL Research University.

- **Jules Merigot** (myself)
  - Masters student in the IASD Master Program (AI Systems and Data Science) at PSL Research University.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

**Note:**

This project is part of ongoing research and is subject to certain restrictions. Please refer to the license section and the [LICENSE.md](LICENSE.md) file for details on how to use this code for research purposes.
