function fmeaavg = computeAcc(changeFile, angleFile)

allchange = load(changeFile);
allslope = load(angleFile);

predlbl = zeros(1,size(allslope,1));
median(allslope)
for i = 1:size(allslope,1)
  if allslope(i) > median(allslope) + std(allslope) 
    predlbl(i) = 1;  
  else
      predlbl(i) = 0;
  end

end

cm2 = confusionmat(allchange,predlbl');
nclass = 2;

% Calculate F-measure
for x=1:nclass

tp = cm2(x,x);
tn = cm2(1,1);
for y=2:nclass
tn = tn+cm2(y,y);
end
tn = tn-cm2(x,x);

fp = sum(cm2(:, x))-cm2(x, x);
fn = sum(cm2(x, :), 2)-cm2(x, x);
pre(x)=tp/(tp+fp+0.01);
rec(x)=tp/(tp+fn+0.01);
fmea(x) = 2*pre(x)*rec(x)/(pre(x)+rec(x)+0.01);
end

fmeaavg = mean(fmea);
dlmwrite('fmeasure.txt',fmea);

end
