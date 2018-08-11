<<<<<<< HEAD
# Introduction
Welcome to the technical interview. Thank you for your interest in being part of our team. This will be divided into 2 parts:

1. A recorded video from you talking about your experience
2. A practical test (code challenge)

# About you
We want to know more about you. So we created some questions and we would like you to record yourself answering those questions and send us back the video file. For that, we recommend this website: https://webcamera.io/. With it, you will be able to record yourself and save the video into a file. This will video file should be sent together with your code (described in the sections below). We **will not**, ever, publish or share your video with anyone. This will only be seen by the technical reviewer assigned to your application.

If you *refuse* to record yourself but still want to participate in our process you may only record an audio of you answering the questions below. For audio recording, you may find a lot of options on the web.

## Questions
The questions we would like you to answer while recording yourself are (please answer in the same order):

1. What is your experience with machine learning development in production?
2. What is your familiarity with Python 3?
3. Can you briefly describe your experience with NumPy, Pandas and other scientific Python libraries
4. Do you have any experience with Scikit Learn or Keras?
5. What is your experience with NLP? Do you know libraries such as NLTK and Spacy?
6. Can you tell the difference between Precision, Recall and F1 score?
7. How do you evaluate your classification models?
8. Which classification algorithms do you have most familiarity?
9. Whats your experience with prediction (regression)?
10. What are the advantages and disadvantages of neural networks?
11. Which clustering algorithms do you have most familiarity?
12. Can you describe your workflow when developing a new model with a new dataset?
13. Do you have experience with Spark or big data tools for machine learning?
14. Can you give examples of scenarios where machine learning is not the best option and where do you think we should apply machine learning?

After recording yourself and downloading the video file, please, watch it and ensure everything is ok. We will declassify you if the video is inaudible or corrupted.

# Code Challenge

## Getting started
Before you start working on the code:

1. Log in into your Bitbucket account or create a new one
2. Fork this repository into a **private repository** (to protect your privacy) [How to fork](https://confluence.atlassian.com/bitbucket/forking-a-repository-221449527.html)
3. Clone your forked version of this repository [How to clone](https://confluence.atlassian.com/bitbucket/clone-a-repository-223217891.html)
4. Be sure that you are using Python version >= 3.6

## Challenge
Find more about your challenge in the `challenge.ipynb` file. If Jupyter (IPython) notebooks are not familiar to you, read more at the [Jupyter Notebook docs](https://jupyter-notebook.readthedocs.io/en/stable/).

## Wrapping up.
1. Be sure that your outputs (described in the last section of the `challenge.ipynb` file) are finished.
2. Commit your changes. [How to commit](https://confluence.atlassian.com/get-started-with-bitbucket/push-code-to-bitbucket-861185309.html)
3. Don't forget to add your video file!
4. Push your changes. [How to push](https://confluence.atlassian.com/get-started-with-bitbucket/push-code-to-bitbucket-861185309.html)
5. At your repository's page: go to _Settings_, then _User and group access_ and add a new user `marcellus777` with _read_ permission.

Thank you!
=======
**Edit a file, create a new file, and clone from Bitbucket in under 2 minutes**

When you're done, you can delete the content in this README and update the file with details for others getting started with your repository.

*We recommend that you open this README in another tab as you perform the tasks below. You can [watch our video](https://youtu.be/0ocf7u76WSo) for a full demo of all the steps in this tutorial. Open the video in a new tab to avoid leaving Bitbucket.*

---

## Edit a file

You’ll start by editing this README file to learn how to edit a file in Bitbucket.

1. Click **Source** on the left side.
2. Click the README.md link from the list of files.
3. Click the **Edit** button.
4. Delete the following text: *Delete this line to make a change to the README from Bitbucket.*
5. After making your change, click **Commit** and then **Commit** again in the dialog. The commit page will open and you’ll see the change you just made.
6. Go back to the **Source** page.

---

## Create a file

Next, you’ll add a new file to this repository.

1. Click the **New file** button at the top of the **Source** page.
2. Give the file a filename of **contributors.txt**.
3. Enter your name in the empty file space.
4. Click **Commit** and then **Commit** again in the dialog.
5. Go back to the **Source** page.

Before you move on, go ahead and explore the repository. You've already seen the **Source** page, but check out the **Commits**, **Branches**, and **Settings** pages.

---

## Clone a repository

Use these steps to clone from SourceTree, our client for using the repository command-line free. Cloning allows you to work on your files locally. If you don't yet have SourceTree, [download and install first](https://www.sourcetreeapp.com/). If you prefer to clone from the command line, see [Clone a repository](https://confluence.atlassian.com/x/4whODQ).

1. You’ll see the clone button under the **Source** heading. Click that button.
2. Now click **Check out in SourceTree**. You may need to create a SourceTree account or log in.
3. When you see the **Clone New** dialog in SourceTree, update the destination path and name if you’d like to and then click **Clone**.
4. Open the directory you just created to see your repository’s files.

Now that you're more familiar with your Bitbucket repository, go ahead and add a new file locally. You can [push your change back to Bitbucket with SourceTree](https://confluence.atlassian.com/x/iqyBMg), or you can [add, commit,](https://confluence.atlassian.com/x/8QhODQ) and [push from the command line](https://confluence.atlassian.com/x/NQ0zDQ).
>>>>>>> 8d485d9235856a4363a75831c59bb2421094fa9a


# Using my Python Module

from challenge import Data, MyKNN  
data = Data("data.csv")  
model = MyKNN(data.X, data.y)  
model.train()  
model.predict()  
model.display_acc()  
model.display_confusion_matrix()  
