program omp_mpi
    use omp_lib
    !use mpi
    implicit none
    ! integer :: ierr, rank, procnum
    integer :: threadnum, id, ierr1, threads, i
    threadnum = 100
    ! rank = 0
    ! call mpi_init(ierr)
    ! call MPI_COMM_RANK(MPI_COMM_WORLD,rank,ierr)
    ! call MPI_COMM_SIZE(MPI_COMM_WORLD,procnum,ierr)
    ! call omp_set_num_threads(threadnum)
    !!$OMP parallel do private(id, ierr1) shared(threadnum, rank, threads)
    !$OMP parallel do private(id) shared(threadnum, threads)
    do i = 1, threadnum
        id = omp_get_thread_num()
        threads = omp_get_num_threads()
        ! print*,id, 'of', threads, 'on CPU: ', rank
        print*,id, 'of', threads
    end do
    !$OMP end parallel do
    ! call mpi_finalize(ierr)
end program