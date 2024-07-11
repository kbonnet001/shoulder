def compute_row_col(x, y) : 
    
    row = (x+y)//4
    if (x+y)%4 != 0 :
        row+=1
    
    return row, min(x+y, 4)