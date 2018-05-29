clc

s = load('dbworld_bodies');
mat = cellstr(s.dictionary); % reading words in to matrix
mat= mat';

word_mat = struct([]);
% char word_mat = zeros([1 4702]); % for comparing words with inputs
% 
% email_mat = struct([]); % for individual emails

filePh = fopen('msg1.txt','w');

%This loop is written for individually writing one text file
%for writing multiple, the outer loop can be run multiple times
for i=1 : 1 : 1
       for j=1 : 1 : 4702
         
           if (s.inputs(i,j)==1)
               %word_mat(i)= mat(1,j);
                %temp = mat(1,j);
                
                %filePh = fopen('msg1.txt', 'a+');
                %fprintf(filePh,'%d %d %d',mat{1,j});
                
                %dlmwrite('msg1.txt',mat{1,j},'-append','delimiter',' ')
                %dlmwrite('msg1.txt',mat{1,j},'delimiter',' ')
                %dlmwrite('msg1.txt',mat{1,j},' ')
                fprintf(filePh,'%s',mat{1,j});
                
                %fid = fopen('msg1.txt','wt');
                % fprintf(fid, '%s', string1, string2, string3);
                
                
           end
           
       end
end

fclose(filePh);
    

