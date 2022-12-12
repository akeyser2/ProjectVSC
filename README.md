Alex Keyser
Data Science 222
December 11, 2022


Welcome to my Data Science final project! This is a project that I am extremely passionate about as I have used data from one of my favorite games. There are many terms in this game that aren't common knowledge that I will go over in this read me. In this project, I clean the data of all of the files, analyse the data from tables and graphs, do multiple hypothysis analysis, and finally use machine learning to predict the outcome of some of my older matches. I am so excited about this project, and I hope you enjoy it as much as I did.

First of all, League of Legends is a 5v5 online multiplayer game. It has multiple types of game that you can play, but most commonly Ranked and Normals. There are 150+ available champions that you can play in the game, as well as the same for your opponents. With this in mind, every single game is completely different from just the champions being played. This does not take into account the skill and decisions of the players themselves. Like I mentioned earlier, there are some terms that I want to briefly go over here because I will be using them in the rest of the project

* Gamemodes
    * There are multiple game modes in LoL, but the ones I've recorded are Ranked, Norms, and ARAM
    * Norms is the baseline of the game, it is also known as Draft sometimes.
    * Ranked is the competitive version of Norms, the exact same game, and playstyle, but taken more seriously
    * ARAM is usually known as the "for fun" mode, you don't get to choose your champion, and there are no rolls like the other game modes
* Roles
    * Norms and Ranked have different roles you can play which is an essential attribute in my data. There are 5 roles; Top, Jungle, Middle, ADC, and Support
    * All roles have different objectives as well as different locations in the map, but this shouldn't be too important
* Other terms
    * KDA: This is your Kill-Death- assist ratio. Usually, the higher it the better you did. It is calculated with this equation
    * CS: This is complicated to explain, but it's a way of tracking how well you were managing multiple things during the game.
    * KP: Kill Participation, this is how many of your Kills or Assists contributed to the team.
    * Average Rank: This is a way of determining the average skill of the lobby. The rank goes from Bronze->Silver->Gold->Plat->etc. As well as from 4 being the lowest to 1 being the highest of that specific rank

I go over these in the main file as well.

In this repository, you will find 6 files other than this README. 

3 CSV files -
    All my ranked games from the past year
    The past 104 games of my Norms and other gamemodes
    General statistics about every champ in the game

Python Notebook -
    The main file of the project

Python file
    Has all of the functions that I call in lol_project.ipynb


With all this information, you are equipped to get started in my project! Take into account that some of my utils calls require information from previous funtions, so when runnning the program, run all.