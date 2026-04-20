function dinput = filter_input(input)

winput=wiener2(input);
dinput=0.1*(winput/max(winput))+0.1;

end