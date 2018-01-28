Code structure:
The files are provided by the teachers.
The only file the student can modify is student_func.cu and it contains the instructions on how to complete the homework. 

Instructions on how to invoke the program:
- Create a directory inside the code, e.g. Make
- Inside the directory inkove: 
	- cmake ..
	- make
- Now you can execute the program typing, inside the directory Make,

 ./HW4 ../red_eye_effect_5.jpg ../red_eye_effect_template_5.jpg 

All the student-provided logic is in student_func.cu. The only other change was done to main.cpp to print also the timing of the reference code.
