# Contributing to rlpyt

Thanks for considering contributing to rlpyt!

It's a research code base, so we'll try to keep things light and easy.  A typical workflow is to start a discussion about bugs or new features in an Issue, and work together to choose a strategy for a fix or enhancement.  Pull Requests are appreciated to lighten my workload, but in some cases I can also just apply changes and push directly.  Please try to make PRs small and focused so they are easy to review.  

## Contributions we're looking for

* Any bug catches/fixes.
* Well-established algorithms which are not yet implemented (we have some compute capacity to use for running verification learning curves).
* Enhancements for wider compatibility on different systems.
* Enhancements for easier use/logging/integrations/workflow.
* Clarifications to documentation.
* Possibly user-contributed example scripts.
* Other ideas?

## Contributions we might not be looking for but could consider

* Extensive customization to your own needs which aren't general to the RL audience (but we can talk about it!).
* Custom environments which are not yet widely used benchmarks.
* Depending on the circumstances, maybe we can make a contrib folder to put these sorts of works into, while keeping the main install light-weight.

## Other notes

Code formatting is mostly according to PEP8 (checked with flake8) but ignoring E128 (indentations).  Note that some keyword arguments are in CamelCase, which is intended to indicate that the argument should be a class or factory function returning some object instance when called.

If you're adding a new functionality that merits a quick unit test, please try to include that under the tests folder.  Right now it's pretty sparse, we might organize it later.

If your bug or issue is to do with a specific RL problem you're working on rather than something to do with the infrastructure in rlpyt, then I might not be able to address it, and it could be lower priority.  But maybe someone in the community can still help.  :)
