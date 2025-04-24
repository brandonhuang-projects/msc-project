function ix = findTime(indx, s2, time)
        [m2,ix_loc] = max(s2(indx));
        ix(1,2) = time(indx(ix_loc));
        
        ix(1,1) = ix_loc;
        
%         ix(1,3) = s2(indx(1));
%         ix(1,4) = s2(indx(2));
%         
%         ix(1,5) = mean(s2(indx));
%         ix(1,6) = sum(s2(indx));
    end