# Imports
import numpy as np
import pandas as pd
import random as rd
import copy as cp
import time
import pickle



# Function to take boolean input
# arg
#   prompt -- string to display for boolean input
# returns
#   user_input -- boolean that the user entered
def takeBooleanInput(prompt):
    prompt += " (y/n)"

    while True:
        user_input = input(prompt)
        if user_input in ('y', 'Y'):
            return True
        elif user_input in ('n', 'N'):
            return False
        else:
            print("Invalid Input! ", end='')
    # End while


# End of function

# Function to take integer input
# arg
#   prompt -- string to display for integer input
# returns
#   user_input -- number that the user entered
def takeIntegerInput(prompt):
    while True:
        user_input = input(prompt)

        if user_input.isnumeric():
            return int(user_input)
    # End while
# End of function


# function to return key for any value
# in the given dictionary
def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key
    # End for

    return None
# End of function

# function to get the uppermost power of 2 near the
# number num
def get_nearest_power_of_2(num):
    power = 0
    x = pow(2, 0)
    while x < num:
        power += 1
        x = pow(2, power)
    return power


class BinaryEncoder:
    def __init__(self, number=0):
        self.number = number
        self.encoded_data_length = 0

        self.zeroes_mask = 0b0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
        self.wildcard_mask = 0b1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111

    # End of function

    # function to fill data
    # arguments
    #   data -- the number to encode
    #   num_bits -- the number of bits in which to encode the data
    # NOTE: remember to fillData first before retrieving anything
    def fillData(self, data, num_bits):
        self.number = self.number << num_bits
        self.number = self.number | data
        self.encoded_data_length += num_bits

    # End of function

    def getEncodedData(self, start_pos, end_pos):
        """
        Get sub binary string
        :param start_pos: Inclusive start position, counting up from MST
        :param end_pos: Non-Inclusive end position
        :return: Binary Sub-String
        """

        temp_num = self.number
        temp_num = temp_num >> self.encoded_data_length - end_pos

        mask = (2 << (end_pos - start_pos - 1)) - 1
        return temp_num & mask

    # End of function

    def getLen(self):
        return self.encoded_data_length

    # End of function

    # Function to change the bit from a specific position
    # arguments
    #   position -- the position from the left most side, used as array indexing
    #   bit -- either 0 or 1 to change at the position
    def modifyBit(self, position, bit):
        assert (position < self.encoded_data_length and (bit == 0 or bit == 1))
        p = self.encoded_data_length - position - 1
        mask = 1 << p
        self.number = (self.number & ~mask) | ((bit << p) & mask)

    # End of function

    def getNumber(self):
        return self.number
    # End of function
# End of class

# Small Test case
testEncoder = BinaryEncoder()
testEncoder.fillData(6, 8)
testEncoder.modifyBit(2, 1)
print (testEncoder.getNumber())


# Function to print the statistics of the dataset
def dataset_statistics(courses_df, teachers_df, students_df, registrations_df):
    print(f"Number of teachers: {len(teachers_df)}")
    print(f"Number of courses: {len(courses_df)}")
    print(f"Number of students: {len(students_df)}")
    print(f"Number of registrations: {len(registrations_df)}")

    print(
        f"\nStudents per course:\n{registrations_df.groupby('courseCode').agg('count').rename(columns={'studentName': 'Number of Registered Students'})}")


# Function to load dataset: arguments = path to the files
# returns pandas dataframes to be used by the population generator
def load_dataset(path):
    teachers_df = pd.read_csv(path + 'teachers.csv', header=None, usecols=[0], names=['teacher'])
    courses_df = pd.read_csv(path + 'courses.csv', header=None, usecols=[0, 1], names=['courseCode', 'courseName'])
    registrations_df = pd.read_csv(path + 'studentCourse.csv', header=None, usecols=[1, 2],
                                   names=['studentName', 'courseCode'])
    students_df = pd.read_csv(path + 'studentNames.csv', header=None, usecols=[0], names=['studentName'])

    # Remove duplicate students, teachers & courses
    students_df.drop_duplicates(subset='studentName', keep='first', inplace=True)
    teachers_df.drop_duplicates(subset='teacher', keep='first', inplace=True)
    courses_df.drop_duplicates(subset='courseCode', keep='first', inplace=True)
    registrations_df.drop_duplicates(['studentName', 'courseCode'], keep='first', inplace=True)

    students_df["studentID"] = students_df.index + 1
    teachers_df['teacherID'] = teachers_df.index + 1

    if True:
        dataset_statistics(courses_df, teachers_df, students_df, registrations_df)

    return courses_df, teachers_df, students_df, registrations_df


class SchedGeneratorGA:
    # ----------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ----------------------------------------------------------------------------------
    # arguments
    #   max_generations       -- the maximum number of generations to run the genetic algorithm
    #   crossover_probability -- the probability of doing a crossover (the other option is to select the parents)
    #   mutation_probabilty   -- the probability that the selected gene will be mutated or not
    #   population_size       -- the number of chromosomes in this population
    #   reset_threshold       -- the number of consective times a fitness occurs for random restart
    def __init__(self, max_generations=100,
                 crossover_probability=0.8,
                 mutation_probability=0.5,
                 population_size=100,
                 reset_threshold=5):

        # Hyperparameters for GA Algorithm
        self.max_generations = max_generations
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.reset_threshold = reset_threshold

        # Indices of the encoding for ease of access
        self.course_code_index = 0
        self.slot_id_index = 1
        self.room_id_index = 2
        self.teacher_index = 3
        self.students_index = 4

        # Metadata on the problem domain
        self.num_registered = 0
        self.num_students = 0
        self.num_teachers = 0
        self.num_courses = 0
        self.min_slots = 0
        self.rooms = []
        self.per_day_slots = 0
        self.friday_break_slot = 0

        # Data on the problem domain
        self.unique_courses = []
        self.num_students_in_course = {}
        self.student_names_map = {}
        self.student_registered = {}

        # Metadata for Binary Encoding
        self.course_code_to_binary = {}
        self.num_slots_string_length = 8
        self.course_string_length = 0
        self.room_string_length = 0
        self.teachers_string_length = 0

    # ----------------------------------------------------------------------------------
    # End of function

    # ----------------------------------------------------------------------------------
    # UTILITY FUNCTIONS
    # ----------------------------------------------------------------------------------

    # Function to return the number of slots from the encoded
    # chromosome
    def get_num_slots(self, chromosome):
        exam = chromosome[0]
        prev_itr = 0
        itr = self.num_slots_string_length
        return exam.getEncodedData(prev_itr, itr)

    # End of function

    def same_day_slots(self, slot_1, slot_2, per_day_slots):
        return int(slot_1 / per_day_slots) == int(slot_2 / per_day_slots)

    # Function to print a chromosome
    def printChromosome(self, chromosome):
        timetable = self.convert_to_timetable(self, chromosome)
        print("\n\n")
        for i in range(len(timetable)):
            print(timetable[i])

    # End of function

    # Function to convert the chromosome to a
    # timetable
    def convert_to_timetable(self, chromosome):
        timetable = []
        for exam in chromosome:
            timetable.append(self.binary_to_exam(exam))

        timetable = sorted(timetable, key=lambda l: l[1])
        return timetable

    # End of function

    # ----------------------------------------------------------------------------------
    # BINARY CONVERSION FUNCTIONS
    # ----------------------------------------------------------------------------------

    # Function to encode an exam to binary
    # arguments
    #     exam            --  the exam to encode
    #     num_slots       --  the number of slots available to the timetable
    # returns
    #     binary_string   --  a BinaryEncoder object containing the encoded number
    def exam_to_binary(self, exam, num_slots):
        binary_string = BinaryEncoder()
        binary_string.fillData(num_slots, self.num_slots_string_length)
        binary_string.fillData(self.course_code_to_binary[exam[self.course_code_index]], self.course_string_length)
        binary_string.fillData(exam[self.slot_id_index], 5)
        binary_string.fillData(exam[self.room_id_index], self.room_string_length)
        binary_string.fillData(exam[self.teacher_index], self.teachers_string_length)
        for i in range(self.num_students):
            if (i + 1) in exam[self.students_index]:
                binary_string.fillData(1, 1)
            else:
                binary_string.fillData(0, 1)
            # End if
        # End for
        return binary_string

    # End of function

    # Function to convert the binary_string back to exam
    # arguments
    #     binary_string   --  the string to decode
    # returns
    #     exam            --  a pyhon array of the format [course_code, slot_id, room_id, teacher_id, students = []]
    def binary_to_exam(self, binary_string):
        exam = []
        prev_itr = 0
        itr = self.num_slots_string_length

        # 1. Converting Num Slots
        num_slots = binary_string.getEncodedData(prev_itr, itr)

        # 2. Converting Course Code
        prev_itr = itr
        itr += self.course_string_length
        course_string = binary_string.getEncodedData(prev_itr, itr)
        course = get_key(self.course_code_to_binary, course_string)
        if course is not None:
            exam.append(course)
        else:
            exam.append(self.unique_courses[rd.randint(0, self.num_courses - 1)])

        # 3. Converting Slot ID
        prev_itr = itr
        itr += 5
        slot = binary_string.getEncodedData(prev_itr, itr)
        if slot < num_slots:
            exam.append(slot)
        else:
            exam.append(rd.randint(0, num_slots - 1))

        # 4. Converting Room ID
        prev_itr = itr
        itr += self.room_string_length
        room = binary_string.getEncodedData(prev_itr, itr)
        if room <= len(self.rooms):
            exam.append(room)
        else:
            exam.append(self.rooms[rd.randint(0, len(self.rooms) - 1)])

        # 5. Converting Teacher ID
        prev_itr = itr
        itr += self.teachers_string_length
        teacher = binary_string.getEncodedData(prev_itr, itr)
        if teacher < self.num_teachers:
            exam.append(teacher)
        else:
            exam.append(rd.randint(0, self.num_teachers - 1))

        # 6. Converting Students
        students = []
        for i in range(self.num_students):
            prev_itr = itr
            itr += 1
            student_string = binary_string.getEncodedData(prev_itr, itr)
            if (student_string == 1):
                students.append(i + 1)
        exam.append(students)
        return exam

    # End of function
    # ----------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------
    # POPULATION GENERATION FUNCIONS
    # ----------------------------------------------------------------------------------

    # Function to initialize or reset a population: main generation algorithm
    # arguments
    #     courses_df          --  pandas dataframe for courses data
    #     teachers_df         --  pandas dataframe for teachers data
    #     registerations_df   --  pandas dataframe for registerations data
    #     rooms               --  python list of available room numbers (integers)
    #     room_capacity       --  average room capacity
    #     min_slots           --  number of available time slots
    # returns
    #     population          --  BinaryEncoded population with given hyperparameters
    def reset_population(self, courses_df, teachers_df, students_df, registrations_df, rooms, room_capacity, min_slots,
                         improved_one=False):
        population = []

        # Setting Initial Variables
        num_rooms = len(self.rooms)
        avg_room_capacity = room_capacity
        num_registrations = self.num_registered
        min_exams_required = int(num_registrations / avg_room_capacity) + 1
        num_course_per_population = int(self.population_size / len(self.unique_courses))

        # Generating the Population
        for i in range(self.population_size):
            chromosome = []

            # Extra exams upto 3x the min number required ideally
            num_exams = min_exams_required + int(rd.uniform(0, 3) * min_exams_required)
            num_slots = min_slots  # + int(rd.uniform(0, 1) * min_slots / 2)

            # Generating Room Slot Pairs
            room_slot_pairs = []
            for slot in range(num_slots):
                for room in range(num_rooms):
                    room_slot_pairs.append([slot, room + 1])
                # End for
            # End for

            # Making temp repos
            courses_repo = cp.deepcopy(self.unique_courses)
            students_repo = cp.deepcopy(self.student_registered)

            # Generating Exams in Chromosome
            if improved_one:
                for course_pick in self.unique_courses:
                    # Selecting Students
                    while len(students_repo[course_pick]) > 0:
                        exam = ["courseCode", "slotID", "roomID", "teacher", "students"]

                        # Selecting Course
                        exam[self.course_code_index] = course_pick

                        # Selecting Room Slot Pair
                        random_slot_room_pair_id = rd.randint(0, len(room_slot_pairs) - 1)
                        exam[self.slot_id_index] = room_slot_pairs[random_slot_room_pair_id][0]
                        exam[self.room_id_index] = room_slot_pairs[random_slot_room_pair_id][1]
                        del room_slot_pairs[random_slot_room_pair_id]

                        # Selecting Teacher
                        exam[self.teacher_index] = int(teachers_df.sample()['teacherID'])

                        # Selecting Students
                        num_students_in_exam = self.num_students_in_course[course_pick]
                        num_students = min(int((0.5 + rd.uniform(0, 1) / 2) * avg_room_capacity), num_students_in_exam)

                        exam[self.students_index] = rd.sample(students_repo[course_pick],
                                                              min(num_students, len(students_repo[course_pick])))
                        for student in exam[self.students_index]:
                            students_repo[course_pick].remove(student)

                        # Converting the Exam to Binary before adding to Chromosome
                        chromosome.append(self.exam_to_binary(exam, num_slots))
                    # End while
                # End for
            else:
                while len(chromosome) < num_exams:
                    exam = ["courseCode", "slotID", "roomID", "teacher", "students"]

                    # Selecting Course
                    course_range = 10
                    course_start = int(i % course_range)
                    course_end = min(course_start + course_range, len(courses_repo) - 1)
                    course_pick = courses_repo[rd.randint(course_start, course_end)]
                    exam[self.course_code_index] = course_pick

                    # Selecting Room Slot Pair
                    random_slot_room_pair_id = rd.randint(0, len(room_slot_pairs) - 1)
                    exam[self.slot_id_index] = room_slot_pairs[random_slot_room_pair_id][0]
                    exam[self.room_id_index] = room_slot_pairs[random_slot_room_pair_id][1]
                    del room_slot_pairs[random_slot_room_pair_id]

                    # Selecting Teacher
                    exam[self.teacher_index] = int(teachers_df.sample()['teacherID'])

                    # Selecting Students
                    if len(students_repo[course_pick]) == 0:
                        students_repo[course_pick] = cp.deepcopy(self.student_registered[course_pick])
                    num_students_in_exam = self.num_students_in_course[course_pick]
                    num_students = min(int((0.5 + rd.uniform(0, 1) / 2) * avg_room_capacity), num_students_in_exam)

                    exam[self.students_index] = rd.sample(students_repo[course_pick],
                                                          min(num_students, len(students_repo[course_pick])))
                    for student in exam[self.students_index]:
                        students_repo[course_pick].remove(student)

                    # Converting the Exam to Binary before adding to Chromosome
                    chromosome.append(self.exam_to_binary(exam, num_slots))
                # End while
            # End If-Else

            # Adding the Chromosome to the Population
            rd.shuffle(chromosome)
            population.append(chromosome)

        # End for

        return population

    # End of function

    # Function to generate population for the first time, setting up the parameters
    # arguments
    #     courses_df          --  pandas dataframe for courses data
    #     teachers_df         --  pandas dataframe for teachers data
    #     registerations_df   --  pandas dataframe for registerations data
    #     rooms               --  python list of available room numbers (integers)
    #     room_capacity       --  average room capacity
    #     min_slots           --  number of available time slots
    # returns
    #     population          --  BinaryEncoded population with given hyperparameters
    def generate_population(self, courses_df, teachers_df, students_df, registrations_df, rooms, room_capacity,
                            min_slots, improved_one=False):

        # Setting Up Parameters for The Class
        self.min_slots = min_slots
        self.rooms = rooms
        self.num_registered = len(registrations_df)
        self.num_students = len(students_df)
        self.num_teachers = len(teachers_df)
        self.room_string_length = get_nearest_power_of_2(len(self.rooms))
        self.teachers_string_length = get_nearest_power_of_2(self.num_teachers)

        # Mapping Course Codes to Binary
        self.unique_courses = list(registrations_df['courseCode'].unique())
        self.num_courses = len(self.unique_courses)
        self.course_string_length = get_nearest_power_of_2(self.num_courses)
        for i, course in enumerate(self.unique_courses):
            self.course_code_to_binary[course] = i

        # Mapping Course Codes to Registered student counts
        student_course_df = registrations_df.groupby(['courseCode']).agg('count').rename(
            columns={'studentName': 'count'}).reset_index()
        for index, row in student_course_df.iterrows():
            self.num_students_in_course[row['courseCode']] = int(row['count'])

        # Mapping student names to student IDs
        for index, row in students_df.iterrows():
            self.student_names_map[row["studentName"]] = int(row['studentID'])

        # Mapping course codes to list of student IDs that are
        # registered
        for index, row in registrations_df.iterrows():
            course_code = row['courseCode']
            if course_code in self.student_registered:
                students_list = self.student_registered[course_code]
                students_list.append(self.student_names_map[row['studentName']])
                self.student_registered[course_code] = students_list
            else:
                students_list = [self.student_names_map[row['studentName']]]
                self.student_registered[course_code] = students_list
            # End If-else
        # End for

        # Generating the Population
        return self.reset_population(courses_df, teachers_df, students_df, registrations_df, rooms,
                                     room_capacity, min_slots, improved_one=improved_one)

    # End of function
    # ----------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------
    # FITNESS FUNCTIONS
    # ----------------------------------------------------------------------------------

    # Function to calculate the fitness of a single chromosome
    # Checks all hard and soft constraints.
    # arguments
    #     cromosome         --  BinaryEncoded object of the chromosome to calculate fitness of
    #     verbose           --  flag for printing the details of the fitness
    # returns
    #     fitness           --  fitness of the cromosome (integer)
    #     binary_cromosome  --  reencoded BinaryEncoded object of the chromosome
    def calculate_fitness_chromosome(self, cromosome, verbose=False):

        # Initial Fitness
        fitness = 10000

        # Converting to Exams
        chromosome = []
        for exam in cromosome:
            chromosome.append(self.binary_to_exam(exam))

        # Initialize Data for Checking Constraints
        num_slots = self.get_num_slots(cromosome)
        student_exam_slot = [[-1] for i in range(self.num_students + 1)]
        teacher_exam_slot = [[-1] for i in range(self.num_teachers + 1)]
        room_slot_pairs = []
        student_no_exam = cp.deepcopy(self.student_registered)
        courses_with_exams = set()
        papers_in_friday_break_slot = []

        for i, exam in enumerate(chromosome):

            # Courses with Exams
            courses_with_exams.add(exam[self.course_code_index])

            # Students without exams
            course = exam[self.course_code_index]
            l2set = set(exam[self.students_index])
            student_no_exam[course] = [x for x in student_no_exam[course] if x not in l2set]

            # Student Exam slot. where student_exam_slot[i][j] is the jth slot of the ith student
            for j in range(len(exam[self.students_index])):
                student = exam[self.students_index][j]
                if (student_exam_slot[student][0] == -1):
                    student_exam_slot[student][0] = exam[self.slot_id_index]
                else:
                    student_exam_slot[student].append(exam[self.slot_id_index])
            # End for

            # Teacher Slots
            teacher = exam[self.teacher_index]
            if (teacher_exam_slot[teacher][0] == -1):
                teacher_exam_slot[teacher][0] = exam[self.slot_id_index]
            else:
                teacher_exam_slot[teacher].append(exam[self.slot_id_index])

            # Room Slots
            room_slot_pairs.append([i, exam[self.room_id_index], exam[self.slot_id_index]])

            # Friday break Slot check
            if (exam[self.slot_id_index] % self.friday_break_slot) == 0:
                papers_in_friday_break_slot.append(exam[self.course_code_index])

        # End for

        # -----------------------------------------------------
        # Hard Constraint #0: Overlapping Room-Slot pairs
        # -----------------------------------------------------
        # Checking for duplicates
        overlapping_pairs = []
        for i, current_pair in enumerate(room_slot_pairs):
            for j, pair in enumerate(room_slot_pairs):
                if i != j and current_pair[1] == pair[1] and current_pair[2] == pair[2]:
                    overlapping_pairs.append([current_pair[1], current_pair[2]])
                # End if
            # End for
        # End for

        # Penalty
        fitness -= (10 * len(overlapping_pairs))

        # -----------------------------------------------------
        # Hard Constraint #1: Every Course Must have an exam
        # -----------------------------------------------------
        # Check if course code set contains all courses taken
        # by students
        courses_without_exams = []
        for course in self.unique_courses:
            if course not in courses_with_exams:
                courses_without_exams.append(course)
        # End for

        # Penalty
        fitness -= int(len(courses_without_exams) * 4)

        # -----------------------------------------------------
        # -----------------------------------------------------
        # Hard Constraint #2: Student Exam clash
        # -----------------------------------------------------
        # Check if a student has a clash in the exam in the same
        # slot
        student_clashes = []
        student_multiple_in_row = []
        for i in range(len(student_exam_slot)):
            unique_slot = []
            for j in range(len(student_exam_slot[i])):
                if (student_exam_slot[i][j] not in unique_slot):
                    unique_slot.append(student_exam_slot[i][j])
                else:
                    student_clashes.append([i, student_exam_slot[i][j]])
                # End if-else
            # End for
            for slot in unique_slot:
                if (slot + 1) in unique_slot and self.same_day_slots(slot, slot + 1, self.per_day_slots):
                    student_multiple_in_row.append([i, slot, slot + 1])
            # End for
        # End for

        # Penalty
        fitness -= int(len(student_clashes) * 1.5)

        # -----------------------------------------------------
        # Hard Constraint #3: Teacher Exam clash
        # -----------------------------------------------------
        # Check whether a teacher has a clash in the exam slots
        teacher_clashes = []
        multiple_in_row = []

        for i in range(len(teacher_exam_slot)):
            unique_slot = []

            for j in range(len(teacher_exam_slot[i])):
                if (teacher_exam_slot[i][j] not in unique_slot):
                    unique_slot.append(teacher_exam_slot[i][j])
                else:
                    teacher_clashes.append([i, teacher_exam_slot[i][j]])
                # End If-Else
            # End for

            for slot in unique_slot:
                if (slot + 1) in unique_slot and self.same_day_slots(slot, slot + 1, self.per_day_slots):
                    multiple_in_row.append([i, slot, slot + 1])
        # End for

        # Penalty
        fitness -= (len(teacher_clashes) * 2)

        # -----------------------------------------------------
        # Hard Constraint #4: Teacher Exam clash
        # -----------------------------------------------------
        # Check whether a teacher is invigilating multiple
        # exams in a row

        # Penalty
        fitness -= (len(multiple_in_row) * 2)

        # -----------------------------------------------------
        # Hard Constraint #5: Student must have every Exam
        # -----------------------------------------------------
        # Check whether every student has an exam of their
        # courses registered

        # Penalty
        for course in self.unique_courses:
            fitness -= int(len(student_no_exam[course]) * 2.25)

        # -----------------------------------------------------
        # Soft Constraint #0: Unused Rooms
        # -----------------------------------------------------
        # Check whether a slot has unused rooms and penalize
        # it
        unused_rooms = []
        for slot in range(num_slots):
            rooms = 0
            for pair in room_slot_pairs:
                if slot == pair[2]:
                    rooms += 1
            # End for
            unused = len(self.rooms) - rooms
            unused_rooms.append(unused)

            # Penalty
            fitness -= int((1 / 2) * unused)

        # End for

        # -----------------------------------------------------
        # Soft Constraint #1: Student exam in a row
        # -----------------------------------------------------
        # Check whether a student has multiple exams in a
        # row
        fitness -= int(len(student_multiple_in_row) * (1 / 2))

        # -----------------------------------------------------
        # Soft Constraint #2: Exam on Friday slot
        # -----------------------------------------------------
        fitness -= int(len(papers_in_friday_break_slot))

        # -----------------------------------------------------
        # Soft Constraint #3: Faculty 2 hour Break
        # -----------------------------------------------------
        faculty_meeting_possible = None
        num_breaks = self.per_day_slots - 1
        break_time = int((8 % (self.per_day_slots * 3)) / num_breaks)

        # Penalty
        if break_time <= 2:
            fitness -= int(break_time * 2)
            faculty_meeting_possible = 'YES'
        else:
            faculty_meeting_possible = 'NO'

        # --------------------------------------------------------
        # Wrapping Constraints
        # --------------------------------------------------------
        constraints_satisfied = None
        if (verbose):
            constraints_satisfied = {}

            # Prepraing dictionary
            constraints_satisfied['overlapping_pairs'] = overlapping_pairs
            constraints_satisfied['courses_without_exams'] = courses_without_exams
            constraints_satisfied['student_no_exam'] = student_no_exam
            constraints_satisfied['student_clashes'] = student_clashes
            constraints_satisfied['teacher_clashes'] = teacher_clashes
            constraints_satisfied['student_multiple_in_row'] = student_multiple_in_row
            constraints_satisfied['papers_in_friday_break_slot'] = papers_in_friday_break_slot
            constraints_satisfied['multiple_in_row'] = multiple_in_row
            constraints_satisfied['unused_rooms'] = unused_rooms
            constraints_satisfied['faculty_meeting_possible'] = faculty_meeting_possible
        # End if

        # Reencoding
        binary_cromosome = []
        for exam in chromosome:
            exam_binary_string = self.exam_to_binary(exam, num_slots)
            binary_cromosome.append(exam_binary_string)
        # End for

        return fitness, binary_cromosome, constraints_satisfied

    # End of function

    # Function to calculate fitness of the entire population
    # arguments
    #     population          --  the population for which the fitness is calculated on
    # returns
    #     population_fitness  --  python array of population fitness, where population_fitness[i]
    #                             is the fitness of the chromosome i
    def calculate_fitness(self, population):
        population_fitness = []
        best = None
        for cromosome in population:
            fitness, best, temp = self.calculate_fitness_chromosome(cromosome)
            cromosome = best
            population_fitness.append(fitness)
        # End for

        return population_fitness

    # End of function

    # ----------------------------------------------------------------------------------
    # PARENT SELECTION FUNCTIONS
    # ----------------------------------------------------------------------------------

    # Function for parent selection using roullette wheel selection
    # arguments
    #     fitness     --  python array of population fitness, where population_fitness[i]
    #                     is the fitness of the chromosome i
    #     population  --  the population from which to select the parents
    # returns
    #     parents     --  selected parents similar to population provided
    def parent_selection(self, fitness, population):
        parents = []

        # Sum of fitness
        total_sum = 0
        for fit in fitness:
            total_sum += fit

        # Roulette Wheel Selection
        temp_fitness = sorted(fitness, reverse=True)
        while len(parents) < len(population):
            marker = rd.uniform(0, total_sum)
            i = 0
            while marker < total_sum:
                marker += fitness[i]
                i += 1
            # End while
            i -= 1
            for j in range(len(fitness)):
                if fitness[j] == fitness[i]:
                    parents.append(population[j])
                    break
                # End if
            # End for
        # End while
        return parents

    # End of function

    def find_two_fittest_individuals(self, fitness, population):
        highest_index = -1
        second_highest_index = -1
        highest_value = 0
        second_highest_value = 0

        for i in range(len(fitness)):
            if fitness[i] > highest_value:
                second_highest_value = highest_value
                second_highest_index = highest_index
                highest_value = fitness[i]
                highest_index = i

            if fitness[i] > second_highest_value and fitness[i] < highest_value:
                second_highest_value = fitness[i]
                second_highest_index = i
        # End for

        return highest_index, second_highest_index

    # End of function

    # ----------------------------------------------------------------------------------
    # CROSSOVER FUNCTIONS
    # ----------------------------------------------------------------------------------

    def apply_crossover_single_point(self, population, parent_a, parent_b):
        cromosome_a = []
        cromosome_b = []
        stop_1 = rd.randint(0, min(len(population[parent_a]), len(population[parent_b])))
        for i in range(stop_1):
            cromosome_a.append(population[parent_a][i])
            cromosome_b.append(population[parent_b][i])
        for i in range(stop_1, len(population[parent_b])):
            cromosome_a.append(population[parent_b][i])

        for i in range(stop_1, len(population[parent_a])):
            cromosome_b.append(population[parent_a][i])
        return cromosome_a, cromosome_b

    def apply_crossover_chromosome(self, population, parent_a, parent_b):
        cromosome_a = []
        cromosome_b = []

        if rd.randint(0, 100) <= 70:
            return self.apply_crossover_single_point(population, parent_a, parent_b)
        # End if

        # CHROMOSOME A
        for exam in population[parent_a]:
            # Searching for common Exam
            common_exam_index = 10000
            exam_string_a = exam.getEncodedData(self.num_slots_string_length,
                                                self.num_slots_string_length + self.course_string_length)
            for i, exam_b in enumerate(population[parent_b]):
                exam_string_b = exam_b.getEncodedData(self.num_slots_string_length,
                                                      self.num_slots_string_length + self.course_string_length)
                if (exam_string_a == exam_string_b):
                    common_exam_index = i
                    break
            # End for
            gene_a = BinaryEncoder()

            if common_exam_index < len(population[parent_a]):
                exam_b = population[parent_b][common_exam_index]

                for i in range(exam.getLen() - self.num_students):
                    gene_a.fillData(exam.getEncodedData(i, i + 1), 1)

                for i in range(exam.getLen() - self.num_students, exam.getLen()):
                    if rd.randint(0, 100) <= 80:
                        gene_a.fillData(exam_b.getEncodedData(i, i + 1), 1)
                    else:
                        gene_a.fillData(exam.getEncodedData(i, i + 1), 1)
                # End for

                cromosome_a.append(gene_a)
            # End if

        # CHROMOSOME_B
        for exam in population[parent_b]:
            # Searching for common Exam
            common_exam_index = 10000
            exam_string_b = exam.getEncodedData(self.num_slots_string_length,
                                                self.num_slots_string_length + self.course_string_length)
            for i, exam_a in enumerate(population[parent_a]):
                exam_string_a = exam_a.getEncodedData(self.num_slots_string_length,
                                                      self.num_slots_string_length + self.course_string_length)
                if (exam_string_b == exam_string_a):
                    common_exam_index = i
                    break
            # End for
            gene_b = BinaryEncoder()

            if common_exam_index < len(population[parent_b]):
                exam_a = population[parent_a][common_exam_index]

                for i in range(exam.getLen() - self.num_students):
                    gene_b.fillData(exam.getEncodedData(i, i + 1), 1)

                for i in range(exam.getLen() - self.num_students, exam.getLen()):
                    if rd.randint(0, 100) <= 80:
                        gene_b.fillData(exam_a.getEncodedData(i, i + 1), 1)
                    else:
                        gene_b.fillData(exam.getEncodedData(i, i + 1), 1)
                # End for

                cromosome_b.append(gene_b)
            # End if
        # End for

        if len(cromosome_a) == 0 or len(cromosome_b) == 0:
            return self.apply_crossover_single_point(population, parent_a, parent_b)

        return cromosome_a, cromosome_b

    # End of function

    def apply_crossover(self, parent_population):
        crossovered_population = []
        population = cp.deepcopy(parent_population)

        while len(crossovered_population) < self.population_size:
            if (len(population) <= 1):
                population = cp.deepcopy(parent_population)

            if rd.randint(0, 100) <= self.crossover_probability * 100:
                added_a = False
                added_b = False
                parent_a = rd.randint(0, len(population) - 1)
                parent_b = rd.randint(0, len(population) - 1)
                while not added_a and not added_b:
                    cromosome_a, cromosome_b = self.apply_crossover_chromosome(population, parent_a, parent_b)

                    # Applying Mutation
                    cromosome_a = self.apply_mutation(cromosome_a)
                    cromosome_b = self.apply_mutation(cromosome_b)

                    # Checking Fitness
                    parent_a_fitness, best_parent_a, temp = self.calculate_fitness_chromosome(population[parent_a])
                    parent_b_fitness, best_parent_b, temp = self.calculate_fitness_chromosome(population[parent_b])
                    cromosome_a_fitness, best_cromosome_a, temp = self.calculate_fitness_chromosome(cromosome_a)
                    cromosome_b_fitness, best_cromosome_b, temp = self.calculate_fitness_chromosome(cromosome_b)

                    # Adding to Population
                    if (cromosome_a_fitness >= parent_a_fitness and cromosome_a_fitness >= parent_b_fitness):
                        crossovered_population.append(best_cromosome_a)
                        added_a = True
                        if parent_a < len(population):
                            del population[parent_a]

                    if (cromosome_b_fitness >= parent_a_fitness and cromosome_b_fitness >= parent_b_fitness):
                        crossovered_population.append(best_cromosome_b)
                        added_b = True
                        if parent_b < len(population):
                            del population[parent_b]

                # End while
            else:
                fitness = self.calculate_fitness(population)
                highest, second_highest = self.find_two_fittest_individuals(fitness, population)
                crossovered_population.append(population[highest])
                crossovered_population.append(population[second_highest])
                if highest < len(population):
                    del population[highest]
                if second_highest < len(population):
                    del population[second_highest]
            # End if
        # End while
        return crossovered_population

    # End of function

    # ----------------------------------------------------------------------------------
    # MUTATION FUNCTIONS
    # ----------------------------------------------------------------------------------
    def apply_mutation(self, cromosome):
        for exam in cromosome:
            lower_bound = self.num_slots_string_length + self.course_string_length
            upper_bound = lower_bound + 5 + self.room_string_length + self.teachers_string_length
            for i in range(lower_bound, upper_bound):
                if rd.randint(0, 100) <= self.mutation_probability * 100:
                    if exam.getEncodedData(i, i + 1) == 1:
                        exam.modifyBit(i, 0)
                    else:
                        exam.modifyBit(i, 1)
        # End if
        return cromosome

    # End of function

    # ----------------------------------------------------------------------------------------------------
    # MAIN GENETIC ALGORITHM FUNCTION
    # ---------------------------------------------------------------------------------------------------
    def run(self, rooms, avg_room_capacity, min_slots, per_day_slots, dataset_path, improved_one=False):

        # Dataset
        self.per_day_slots = per_day_slots
        self.friday_break_slot = (4 * per_day_slots) + 2
        courses_df, teachers_df, students_df, registrations_df = load_dataset(dataset_path)

        # Generating Population
        population = self.generate_population(courses_df, teachers_df, students_df, registrations_df, rooms,
                                              avg_room_capacity, min_slots, improved_one)

        # Generation 0
        fitness = self.calculate_fitness(population)
        candidate1, candidate2 = self.find_two_fittest_individuals(fitness, population)
        current_fitness = fitness[candidate1]
        print("Current generation:", 0, " Current Solution:", current_fitness, " Best solution so far:",
              current_fitness)
        best_solution = cp.deepcopy(population[candidate1])
        best_solution_value = current_fitness
        prev_solution = 0
        prev_solution_count = 0

        # Iterating over the generations
        for generation in range(self.max_generations):
            # Genetic Algorithm Main Flow
            parents = self.parent_selection(fitness, population)
            crossovered = self.apply_crossover(parents)
            population = crossovered
            fitness = self.calculate_fitness(population)
            candidate1, candidate2 = self.find_two_fittest_individuals(fitness, population)
            current_fitness = fitness[candidate1]

            # Selecting Best Solution
            if best_solution is None:
                best_solution = cp.deepcopy(population[candidate1])
                best_solution_value = current_fitness

            elif current_fitness > best_solution_value:
                best_solution = cp.deepcopy(population[candidate1])
                best_solution_value = current_fitness

            # Printing
            if generation % 1 == 0:
                print("Current generation:", generation + 1, " Current Solution:", current_fitness,
                      " Best solution so far:", best_solution_value)

            # Random Resetting
            if current_fitness == prev_solution:
                prev_solution_count += 1
            else:
                prev_solution = current_fitness
                prev_solution_count = 1

            if prev_solution_count >= self.reset_threshold:
                population = self.reset_population(courses_df, teachers_df, students_df, registrations_df, rooms,
                                                   avg_room_capacity, min_slots, improved_one)

        # End for

        return best_solution
    # End of function
# End of Class

class Timetable:
    def __init__(self):
        self.scheduleGenerator = None

        # Dataset
        self.min_slots = 0
        self.avg_classroom_size = 0
        self.rooms = []

        self.courses_df = None
        self.teachers_df = None
        self.students_df = None
        self.registrations_df = None
        self.unique_courses = None

        self.exam_duration = 0
        self.num_days = 0
        self.per_day_slots = 0
        self.slot_id_index = 0
        self.exam_times = []

        # Data
        self.time_table = None
        self.best_chromosome = None

    # End of function

    # Function to print the timetable
    def print_timetable(self):
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        prev_day = -1
        num_days = 0
        prev_slot = -1
        for exam in self.time_table:
            current_day = int(exam[self.slot_id_index] / self.per_day_slots) % 5
            if prev_day != current_day:
                num_days += 1
                print("-------------------------------------------------------------")
                print("Day #", num_days, " : ", days[current_day])
                print("-------------------------------------------------------------")
                prev_day = current_day
            slot = exam[self.slot_id_index]
            if prev_slot != slot:
                print("\nStart Time = " + str(self.exam_times[slot % self.per_day_slots][0]) + ":00\tEnd Time = "
                      + str(self.exam_times[slot % self.per_day_slots][1]) + ":00")

                prev_slot = slot
            print("Course:", exam[0], "Room Number:", exam[2], end=' ')
            print("Teacher ID:", exam[3], "Student IDs:", exam[4])
        # End for

    # End of function

    # Function to print the constraints satisfied
    # by the timetable
    def print_constraints_data(self):
        # Calculating fitness
        fitness, string, constraints_satisfied = self.scheduleGenerator.calculate_fitness_chromosome(
            self.best_chromosome, verbose=True)

        # Unpacking Data
        overlapping_pairs = constraints_satisfied['overlapping_pairs']
        courses_without_exams = constraints_satisfied['courses_without_exams']
        student_no_exam = constraints_satisfied['student_no_exam']
        student_clashes = constraints_satisfied['student_clashes']
        teacher_clashes = constraints_satisfied['teacher_clashes']
        student_multiple_in_row = constraints_satisfied['student_multiple_in_row']
        papers_in_friday_break_slot = constraints_satisfied['papers_in_friday_break_slot']
        multiple_in_row = constraints_satisfied['multiple_in_row']
        unused_rooms = constraints_satisfied['unused_rooms']
        faculty_meeting_possible = constraints_satisfied['faculty_meeting_possible']

        # Printing
        print("-------------------------------")
        print("FITNESS = ", fitness)
        print("-------------------------------")

        print("-------------------------------")
        print("HARD CONSTRAINTS")
        print("-------------------------------")
        print("Overlapping Room-Slot Pairs =", *overlapping_pairs, sep=' ')
        print("Courses without Exams =", *courses_without_exams, sep=' ')
        print("Students without Exams =")
        for course in self.unique_courses:
            if len(student_no_exam[course]) > 0:
                print(course, " : ", student_no_exam[course])
        # End for
        print("Students Clash =")
        for i, clash in enumerate(student_clashes):
            print(clash, end=', ')
            if i % 10 == 0 and i != 0:
                print("")
        # End for
        print("\nTeachers Clash =", *teacher_clashes, sep=', ')
        print("Teachers Consective Clash =", *multiple_in_row, sep=', ')

        print("-------------------------------")
        print("SOFT CONSTRAINTS")
        print("-------------------------------")
        print("Unused Rooms = ", *unused_rooms, sep=' ')
        print("Student Consective Clash =", *student_multiple_in_row, sep=', ')
        print("Papers in Friday break slot =", *papers_in_friday_break_slot, sep=', ')
        print("Is Faculty Meeting Possible =", faculty_meeting_possible)

    # End of function

    # Function for slot-related calculations
    def calculate_slots(self, exam_duration, num_days):
        self.per_day_slots = int(8 / exam_duration)
        self.min_slots = self.per_day_slots * num_days

        # Storing exam times
        num_breaks = self.per_day_slots - 1
        break_time = int((8 % (self.per_day_slots * exam_duration)) / num_breaks)
        start_time = 9
        end_time = start_time + exam_duration
        for i in range(self.per_day_slots):
            self.exam_times.append([start_time, end_time])
            start_time = (end_time + break_time)
            end_time = start_time + exam_duration

    # Main Function to Generate the timetable
    def generate_timetable(self, rooms, avg_classroom_size, exam_duration,
                           num_days, dataset_path, best_one=False):
        # Processing Dataset
        print("Processing Dataset ...")
        self.rooms = rooms
        self.courses_df, self.teachers_df, self.students_df, self.registrations_df = load_dataset(dataset_path)
        self.unique_courses = list(self.registrations_df['courseCode'].unique())
        self.exam_duration = exam_duration
        self.avg_classroom_size = avg_classroom_size
        self.num_days = num_days

        # Calculating Slots
        self.calculate_slots(exam_duration, num_days)

        # Setting Hyperparameters
        max_generations = 5
        crossover_probability = 0.8
        mutation_probability = 0.5
        population_size = 5
        if best_one:
            max_generations = 0

        # Running GA
        print("Running Genetic Algorithm ...")
        self.scheduleGenerator = SchedGeneratorGA(max_generations=max_generations,
                                                  crossover_probability=crossover_probability,
                                                  mutation_probability=mutation_probability,
                                                  population_size=population_size)
        self.best_chromosome = self.scheduleGenerator.run(rooms, avg_classroom_size, self.min_slots, self.per_day_slots,
                                                          dataset_path, best_one)

        # Setting Post Op Data
        self.slot_id_index = self.scheduleGenerator.slot_id_index

        # Converting and Storing
        time_table = self.scheduleGenerator.convert_to_timetable(self.best_chromosome)
        self.time_table = time_table

    # End of function
# End of class

# Parameters
dataset_path = 'dataset/'
rooms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
avg_room_capacity = 28
exam_duration_in_hours = 3
max_days = 10
file_path = 'time_table_1.pkl'

# Generating the Time table
timeTable = Timetable()
timeTable.generate_timetable(rooms, avg_room_capacity, exam_duration_in_hours, max_days, dataset_path)

# Printing Constraints Satisfied
timeTable.print_constraints_data()

# Printing the Time Table
timeTable.print_timetable()

# Saving to file
with open(file_path, 'wb') as output:
    pickle.dump(timeTable, output, pickle.HIGHEST_PROTOCOL)

# Loading from file
file_path = 'time_table_1.pkl'
with open(file_path, 'rb') as input:
    timeTable = pickle.load(input)
    timeTable.print_timetable()