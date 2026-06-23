import streamlit as st


from src.rag_core import run_rag_pipeline



st.title(

"CrediTrust Complaint Assistant"

)



question = st.text_input(

"Ask a question"

)




if st.button("Ask"):



    answer,sources=run_rag_pipeline(

        question

    )



    st.subheader(

        "Answer"

    )


    st.write(

        answer

    )



    st.subheader(

        "Sources"

    )



    for s in sources:


        st.write(s)




if st.button(

"Clear"

):


    st.rerun()