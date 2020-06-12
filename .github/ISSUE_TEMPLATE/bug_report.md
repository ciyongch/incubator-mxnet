<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->
<!--- -->
<!---   http://www.apache.org/licenses/LICENSE-2.0 -->
<!--- -->
<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

---
name: Bug report
about: Create a report to help us improve
title: ''
labels: 'Bug'
assignees: ''

---
## Description
(A clear and concise description of what the bug is.)
(This is a new line added....)

### Error Message
(Paste the complete error message. Please also include stack trace by setting environment variable `DMLC_LOG_STACK_TRACE_DEPTH=10` before running your script.)

## To Reproduce
(If you developed your own code, please provide a short script that reproduces the error. For existing examples, please provide link.)

### Steps to reproduce
(Paste the commands you ran that produced the error.)

1.
2.

## What have you tried to solve it?

1.
2.

## Environment

We recommend using our script for collecting the diagnositc information. Run the following command and paste the outputs below:
```
curl --retry 10 -s https://raw.githubusercontent.com/dmlc/gluon-nlp/master/tools/diagnose.py | python

# paste outputs here
```
