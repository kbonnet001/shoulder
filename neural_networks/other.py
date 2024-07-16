# def compute_row_col(x, y, div) : 
    
#     row = (x+y)//div
#     if (x+y)%div != 0 :
#         row+=1
    
#     return row, min(x+y, div)

def compute_row_col(sum, div) : 
    """Compute ideal row and col for subplots
    
    INPUTS : 
    - sum : int, sum of all number of plot to do in the figure
    - div : int, number of lines max 
    
    OUTPUTS : 
    - row : int, number of row for subplot
    - col : int, number of col for subplot """
    
    row = sum//div
    if sum%div != 0 :
        row+=1
    
    return row, min(sum, div)