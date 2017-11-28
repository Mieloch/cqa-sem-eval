==============================================================
StackExchange data for SemEval-2017 Task 3, Subtask E
"Community Question Answering"
Version 1.0: August 26, 2016
==============================================================

This file contains the basic information regarding the StackExchange data provided for the SemEval-2017 task "Community Question Answering", Subtask E. The current version (1.0, August 26, 2016) corresponds to the release of the sample data. The training and test sets will be provided in future versions. All changes and updates on these data sets are reported in Section 1 of this document.


[1] LIST OF VERSIONS

  v1.0 [2016/26/08]: initial distribution of the StackExchange SAMPLE data:
                     27 original questions, 520 related questions, 838 answers, 1194 comments to questions, and 264 comments to answers


[2] CONTENTS OF THE DISTRIBUTION 1.0

We provide the following files:


* MAIN files

  * README.txt 
    this file

  * stackexchange_sample.xml
    The SAMPLE data consisting of 27 original questions, 520 related questions, 838 answers, 1194 comments to questions and 264 comments to answers; The data comes from the StackExchange Cooking site, which is NOT one of the sites that the TRAINING, DEV or TEST data will be taken from. All annotations come from users of the StackExchange website.



NOTES:

  1. Some of the related questions are repeated as a question can be potentially related to more than one original question.

  2. This distribution is directly downloadable from the official SemEval-2017 Task 3 website:
  http://alt.qcri.org/semeval2017/task3/index.php?id=data-and-tools



[3] LICENSING

  These datasets are free for general research use.


[4] CITATION

  StackExchange needs to be attributed if you use this data.


[5] SUBTASKS

* Subtask E (English): Multi-Domain Duplicate Detection.
  Given a new question (aka the original question) and a set of 50 candidate questions, rerank the 50 candidate questions according to their relevance with respect to the original question, and truncate the result list in such a way that only "PerfectMatch" questions appear in it.


[6] DATA FORMAT

The datasets are XML-formatted and the text encoding is UTF-8.
Below we focus our description on the format of the main files;
we then briefly describe the format of the alternative format files.

A dataset file is a sequence of examples (original questions):

<?xml version="1.0" ?>
  <OrgQuestion> ... </OrgQuestion>
  <OrgQuestion> ... </OrgQuestion>
  ...
  <OrgQuestion> ... </OrgQuestion>
</xml>

Each OrgQuestion has an ID, e.g., <OrgQuestion ORGQ_ID="236203783777">

The structure of an OrgQuestion is the following:

<OrgQuestion ...>
  <OrgQSubject> text </OrgQSubject>
  <OrgQBody> text </OrgQBody>
  <Thread ...>
    <RelQuestion ...> ... </RelQuestion>
    <RelComment ...> <RelCText> text </RelCText> </RelComment>  (any number possible, including zero)
    <RelAnswer ...>  (any number possible, including zero)
        <RelAText> text </RelAText>
        <RelAComment ...> <RelACText> text </RelACText> </RelAComment> (any number possible, including zero)
    </RelAnswer>
  </Thread>
</OrgQuestion>

The text between the <OrgQSubject> and the </OrgQSubject> tags is the subject of the original question (cleansed version, e.g., with new lines removed and some further minor changes).

The text between the tags <OrgQBody> and </OrgQBody> is the main body of the question (again, this is a cleansed version, e.g., with new lines removed and some further minor changes).

What follows is a Thread, which contains one potentially related question RelQuestion, with its answers and comments.


*** Thread ***

A thread has one obligatory attribute as in the following example:

<Thread THREAD_SEQUENCE="236203783777_R605372350975">
where 236203783777 is the ID of the OrgQuestion, and 605372350975 the ID of the RelQuestion in the thread.


*** RelQuestion ***

Each RelQuestion tag has a list of attributes, as in the following example:

<RelQuestion RELQ_CATEGORY="cooking" RELQ_DATE="2015-09-27 15:10:30" RELQ_ID="236203783777_R605372350975" RELQ_RANKING_ORDER="4" RELQ_RELEVANCE2ORGQ="Irrelevant" RELQ_SCORE="1" RELQ_TAGS="deep-frying" RELQ_USERID="33023" RELQ_USERNAME="" RELQ_VIEWCOUNT="1054">

- RELQ_ID: the same as for the thread (as there is a 1:1 correspondence between a RelQuestion and its thread)
- RELQ_RANKING_ORDER: the rank of the related question thread in the list of candidates. This has no particular meaning.
- RELQ_CATEGORY: the StackExchange subforum that the question came from.
- RELQ_DATE: the date and time of posting
- RELQ_SCORE: the sum of the nr of upvotes and downvotes the question has received from the community.
- RELQ_TAGS: the tags the question has received from the community. This is not a fixed set.
- RELQ_VIEWCOUNT: the number of times this question has been viewed by users.
- RELQ_USERID: internal identifier for the user who posted the question; consistent across all questions, answers and comments per StackExchange subforum.
- RELQ_USERNAME: this field is always empty. It's only there for consistency with the other data sets.
- RELQ_RELEVANCE2ORGQ: relevance of the thread of this RelQuestion with respect to the OrgQuestion. This label could be 
  - PerfectMatch: RelQuestion matches OrgQuestion (almost) perfectly. The information needs are very similar.
  - Related: RelQuestion covers some aspects of OrgQuestion, but is not quite a duplicate.
  - Irrelevant: RelQuestion covers no aspects of OrgQuestion.

The structure of a RelQuestion is the following:

<RelQuestion ...>
  <RelQSubject> text </RelQSubject>
  <RelQBody> text </RelQBody>
</RelQuestion>

The text between the <RelQSubject> and the </RelQSubject> tags is the subject of the related question.
The text between tags <RelQBody> and </RelQBody> is the main body of the related question.


*** RelAnswer ***

Each RelAnswer tag has a list of attributes, as in the following example:

<RelAnswer RELA_ACCEPTED="0" RELA_DATE="2015-10-28 03:01:27" RELA_ID="236203783777_R311285158104_A156280127674" RELA_RELEVANCE2ORGQ="" RELA_RELEVANCE2RELQ="" RELA_SCORE="1" RELA_USERID="6170" RELA_USERNAME="">

- RELA_ID: internal identifier of the comment. The first part is the OrgQuestion ID, the second part the RelQuestion ID and the third part is the answer ID.
- RELA_DATE: the date and time of posting
- RELA_USERID: internal identifier of the user posting the comment; consistent across all questions, answers and comments per StackExchange subforum.
- RELA_USERNAME: this field is always empty. It's only there for consistency with the other data sets.
- RELA_RELEVANCE2ORGQ: this field is always empty. It's only there for consistency with the other data sets.
- RELA_RELEVANCE2RELQ: this field is always empty. It's only there for consistency with the other data sets.
- RELA_SCORE: the sum of the nr of upvotes and downvotes the answer has received from the community.
- RELA_ACCEPTED: whether the answer has been accepted as the best one by the person that asked a question, or not. Values: 1, 0.


Answers are structured as follows:

<RelAnswer ...>
  <RelAText> text </RelAText>
  <RelAComment ...> ... </RelAComment> (more info on this below)
</RelAnswer>

The text between the <RelAText> and the </RelAText> tags is the text of the answer.


*** RelComment and RelAComment ***

These are comments to the candidate question or comments to answers.
Each RelComment/RelAComment tag has a list of attributes, as in the following examples:

<RelComment RELC_DATE="2015-10-28 07:40:27" RELC_ID="236203783777_R311285158104_C200752356162" RELC_RELEVANCE2ORGQ="" RELC_RELEVANCE2RELQ="" RELC_SCORE="0" RELC_USERID="4638" RELC_USERNAME="">
<RelAComment RELAC_DATE="2015-09-27 15:53:59" RELAC_ID="236203783777_R311285158104_A132825634416_C216382489616" RELAC_RELEVANCE2ORGQ="" RELAC_RELEVANCE2RELQ="" RELAC_SCORE="0" RELAC_USERID="15020" RELAC_USERNAME="">

- RELC_ID/RELAC_ID: internal identifier of the comment. The first part is the OrgQuestion ID, the second part the RelQuestion ID and the third part is the comment ID.
- RELC_DATE/RELAC_DATE: the date and time of posting
- RELC_USERID/RELAC_USERID: internal identifier of the user posting the comment; consistent across all questions, answers and comments per StackExchange subforum.
- RELC_USERNAME/ERLAC_USERNAME: this field is always empty. It's only there for consistency with the other data sets.
- RELC_RELEVANCE2ORGQ/RELAC_RELEVANCE2ORGQ: this field is always empty. It's only there for consistency with the other data sets.
- RELC_RELEVANCE2RELQ/RELAC_RELEVANCE2RELQ: this field is always empty. It's only there for consistency with the other data sets.
- RELC_SCORE/RELAC_SCORE: the sum of the nr of upvotes and downvotes the comment has received from the community.

 
Comments are structured as follows:

<RelComment ...>
  <RelCText> text </RelCText>
</RelComment>

or

<RelAComment ...>
  <RelACText> text </RelACText>
</RelAComment>

depending on whether it is a comment to a question or a comment to an answer.

The text between the <RelCText> and the </RelCText> (or <RelACText> and </RelACText>) tags is the text of the comment.




[7] MORE INFORMATION ABOUT THE STACKEXCHANGE DATA

The StackExchange data comes from several different StackExchange sites: http://stackexchange.com/

A sample of questions was automatically selected and the 50 candidate duplicate questions were selected semi-automatically and semi-randomly.
In the SAMPLE, TRAIN and DEV data the annotations come directly from the StackExchange communities. As such, the existing labels are of high quality, but there may be some labels missing. The SAMPLE, TRAIN and DEV data should therefore be regarded as a silver standard. For the TEST data all question pairs have been annotated manually, to provide a true gold standard.



[8] STATISTICS

This will be filled with information when the TRAIN, DEV and TEST sets are released.



[9] CREDITS

Task Organizers:

  Preslav Nakov, Qatar Computing Research Institute, HBKU
  Lluís Màrquez, Qatar Computing Research Institute, HBKU
  Alessandro Moschitti, Qatar Computing Research Institute, HBKU
  Timothy Baldwin, The University of Melbourne
  Doris Hoogeveen, The University of Melbourne
  Karin M Verspoor, The University of Melbourne
  

Task website: http://alt.qcri.org/semeval2017/task3/

Contact: semeval-cqa@googlegroups.com

