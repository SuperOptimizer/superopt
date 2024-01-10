# onedaybuilds

OneDayBuilds is a project to test a new approach to software engineering. namely:

1) The day is the singular unit of work.
2) If a task is too large to be accomplished in a single day, split it up into day length tasks.
3) If a task is too small, add it together with other smaller tasks and make the composite the task.
4) One squashed commit is pushed at the end of the day.
5) Code should take the minimum amount of files as necessary, preferrably one per day. E.g. 1 `.c`, `.h`, and `Makefile`
7) Code is write once, read many. If updates are necessary to a previous day's project, copy it to the current day's directory and make the necessary changes.
8) Code must import code from previous days in priortity of more recent days havin higher import priority.
9) The day's solution should include some number of features, an end to end example, and a number of unit tests.

## why?

The driving idea behind onedaybuilds is that the entire plan/develop/build/test/deploy/document cycle can and should be compressed into a single day. This emphases the goal of getting things done, rather than fussing about design or trying to have too many people working on too many disparate parts.
