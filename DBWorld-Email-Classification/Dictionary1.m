clc


files = dir('C:\Users\Home\Desktop\IR Term Project\IR Project\dbworld emails dataset\dictionary_processing\*.txt');
len = length(files);

%C:\Users\Home\Desktop\IR Term Project\IR Project\dbworld emails dataset\dictionary_processing
data = struct([]); 
data{len} = [];

for k = 1:len

data{k} =  textread(fullfile('C:\Users\Home\Desktop\IR Term Project\IR Project\dbworld emails dataset\dictionary_processing\',files(k).name), '%s', 'delimiter', ' ');
if (k == 1)
    temp = data{k};
else
    temp = vertcat(temp,data{k});
end
end
[uv,~,idx] = unique(temp);
n = accumarray(idx(:),1);
[value,index]=sort(n(:),'descend');
words = uv(index);
count = num2cell(value);
dict = [words,count];
