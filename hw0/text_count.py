import sys

# read text file funciton
def read_text(file_name):
	txt_file = open(file_name,'r')
	lines = txt_file.readlines()
	txt_file.close()

	return lines

# read text file
file_name = sys.argv[1]
text = read_text(file_name)

# split text
text_list = text[0].split()

# text count
counted_num = {}
order_list = []
for word in text_list:
	if word in counted_num:
		counted_num[word]+=1

	else:
		counted_num[word] = 1
		order_list.append(word)

# write count num into text
write_file_name = 'Q1.txt'
write_file = open(write_file_name, 'w')

for i in range(len(order_list)):

	if i == len(order_list)-1:
		word = order_list[i]
		write_file.write("%s %d %d" %(word, i, counted_num[word]))
	else:
		word = order_list[i]
		write_file.write("%s %d %d\n" %(word, i, counted_num[word]))

write_file.close()