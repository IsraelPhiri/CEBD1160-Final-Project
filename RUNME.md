A)Building Dockerfile:
 
From a terminal, after verifying that docker is running,(note the space before "." at end of this command) type:
  docker build -t FinalProject .


B)Running Image:

After dockerfile is built, run it by (without "." at the end of command)typing:
  docker run FinalProject

Dockerfile will run and execute the project_processor.py for wine data analysis from sklearn.
