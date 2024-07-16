def compute_row_col(x, y, div) : 
    
    row = (x+y)//div
    if (x+y)%div != 0 :
        row+=1
    
    return row, min(x+y, div)