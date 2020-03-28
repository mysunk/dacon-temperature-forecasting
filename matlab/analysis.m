%%
figure;
plot(Y15,'-')
hold on
plot(Y16,':')
plot(Y18,'--')
legend(['15';'16';'18';]);

%% residual
res = zeros(432,2);
res(:,1) = Y18 - Y15;
res(:,2) = Y18 - Y16;
figure;
plot(res)
hold on
plot(mean(res,2),'k--')
% legend({'residual of Y15, Y18';'residual of Y16, Y18','mean'})

%%
figure;
subplot(3,1,1)
plot(res(1:144,1))
hold on
plot(res(1:144,2))
legend(['residual of Y15, Y18';'residual of Y16, Y18'])
subplot(3,1,2)
plot(res(144+1:144*2,1))
hold on
plot(res(144+1:144*2,2))
legend(['residual of Y15, Y18';'residual of Y16, Y18'])
subplot(3,1,3)
plot(res(144*2+1:144*3,1))
hold on
plot(res(144*2+1:144*3,2))
legend(['residual of Y15, Y18';'residual of Y16, Y18'])


%%
num = 2;
figure;
day1 = res(1:144,num);
day2 = res(144+1:144*2,num);
day3 = res(144*2+1:144*3,num);
plot(day1)
hold on
plot(day2)
plot(day3)
plot(mean([day1,day2,day3],2),'k')
legend(['day 1';'day 2';'day 3';])
%%
% Y18_with_y16 = repmat(mean([day1,day2,day3],2),3,1) + Y16;
% Y18_with_y16 = mean([day1,day2,day3],'all') + Y16;
Y18_with_y15 = mean([day1,day2,day3],'all') + Y15;
figure;
plot(Y18_with_y16)
hold on
plot(Y18)
legend(['aug';'gro'])

%% make new Y18
num = 1;
day1 = res(1:144,num);
day2 = res(144+1:144*2,num);
day3 = res(144*2+1:144*3,num);
Y18_with_y15_1 = repmat(mean([day1,day2,day3],2),30,1) + Y15_1;
mean_res_y15 = repmat(mean([day1,day2,day3],2),80,1);
csvwrite('mean_res_y15.csv',mean_res_y15)


num = 2;
day1 = res(1:144,num);
day2 = res(144+1:144*2,num);
day3 = res(144*2+1:144*3,num);
Y18_with_y16_1 = repmat(mean([day1,day2,day3],2),30,1) + Y16_1;
mean_res_y16 = repmat(mean([day1,day2,day3],2),80,1);
csvwrite('mean_res_y16.csv',mean_res_y16)
%%
figure;
plot(Y18_with_y15_1);
hold on
plot(Y18_with_y16_1);
plot(Y_18_ms_1);
plot(Y18_sw_1);
plot(Y18_sw_2);
legend(['15';'16';'ms';'sw';'sw'])

%%
Y_18_ms = [Y18_with_y15_1,Y18_with_y16_1];
Y_18_ms = [Y_18_ms ;repmat(Y18,1,2)];
Y_18_ms_1 = Y18_with_y15_1*0.73 + Y18_with_y16_1*0.27;
csvwrite('Y18_ms.csv',Y_18_ms)

%% weighted sum
rmse_val = [];
for i=0:100
    result = Y18_with_y16*0.01*i + Y18_with_y15*(1-0.01*i);
    rmse_val = [rmse_val;immse(result, Y18)];
end


%%
res_16_1 = mean(res_16_1);
res_16_2 = mean(res_16_2);
res_16_3 = mean(res_16_3);
% res_16_3 = res_16_3;
res_16_4 = mean(res_16_4);
res_16_5 = mean(res_16_5);

%%
res_16 = zeros(11520,1);
for i=0:79
    for j=0:143
        if mod(j,144) <45
           res_16(i*144+j+1) = res_16_1;
        elseif  mod(j,144) <64
           res_16(i*144+j+1) = res_16_2; 
        elseif  mod(j,144) <88
            res_16(i*144+j+1) = res_16_3; 
           % res_16(i*144+j+1) = res_16_3(mod(j,144)-64+1);
        elseif  mod(j,144) <110
           res_16(i*144+j+1) = res_16_4;
        else
           res_16(i*144+j+1) = res_16_5;
        end
    end
end


%%
csvwrite('res_16.csv',res_16)
csvwrite('res_16.csv',res_16)
%%
figure;
plot(y_pred_1)
hold on
plot(y_pred_2)
plot(sw)
legend(['ms_1';'ms_2';'sw__'])

%%
figure;
plot(res_16)
%%
figure;
plot(Y09)
hold on
plot(Y15)
plot(Y16)
plot(Y18)
legend('1','2','3','4')
%%
figure;
plot(Y15+1.7)
hold on
plot(Y18)

%%
figure;
plot(res_additional)
%%
res_additional =res_additional  + 1.7;
csvwrite('res_15_2.csv',res_15 )
%% 시간대별로 residual을 다르게
res = Y18 - Y15 - 1.7;
res_additional= zeros(144,1);
for i=0:2
    for j=0:143
        if mod(j,144) >= 64 && mod(j,144) <= 86
            res_additional(j) = res_additional(j)+ res(i*144+j+1);
        end
    end
end
res_additional = res_additional/3;

res_15 = repmat(res_additional, 80, 1);

%%
figure;
plot(Y15 + repmat(res_additional,3,1)+1.7);
hold on
plot(Y18);

%%
figure;
plot(Y09+1.7)
hold on
plot(Y18)