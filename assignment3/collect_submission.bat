@echo off
if exist assignment_3_submission.zip del /F /Q assignment_3_submission.zip
tar -a -c -f assignment_3_submission.zip  *.py style_modules/*.py visualizers/*py
