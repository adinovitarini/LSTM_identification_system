%% Comparison Model
nh=5;
[net,traininfo] = LSTM_model(CP1noDis,CP2noDis,nh);
consig = CP1noDis(:,1);
%1 RNN model 
 for i = 1:1000
[~,YPred(i)] = predictAndUpdateState(net,consig(i));
 end
%2 Haykin model
[Model_haykin,x_haykin,y_haykin] = HaykinModel(net,consig);
%3 Linearize Haykin model 
[Model_linearize,x_lin,y_lin] = linearizeModel(net,consig);
%% Plot 
figure(2);clf
plot((CP1noDis(:,2)),'k','LineWidth',2)
hold on
plot(YPred,'b','LineWidth',2)
hold on
plot(y_haykin,'r','LineWidth',2)
hold on
plot(y_lin,'m','LineWidth',2)
grid on
legend('Real','RNN','Haykin model','Linearize Haykin model')
xlabel('time')
title("Method Comparison for Validation Dataset for nh = "+nh+".")