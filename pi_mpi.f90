! program pi_mpi
!     use omp_lib
!     ! use mpi 
!     implicit none
!     real(8) :: n
!     real(8) :: result
!     integer, parameter :: times = 1000000
!     real(4) :: t1,t2
!     result = 0.0
!     t1 = second()
!     !$OMP parallel do private(result)
!     do n = 1,times
!         result = result+(-1)**(n-1)/(2*(n-1)+1)
!     end do
!     !$OMP end parallel do
!     t2 = second()
!     write(*,*) result*4,t2-t1
! end program


program pi_mpi
    use omp_lib
    use mpi 
    implicit none
    real(8) :: n
    real(8) :: result
    integer, parameter :: times = 1000000
    integer :: ierr, rank, procnum
    real(4) :: t1,t2
    call mpi_init(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD,rank,ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD,procnum,ierr)
    result = 0.0
    t1 = second()
    !$OMP parallel do private(result)
    do n = 1+rank*times/2,(rank+1)*times/2
        result = result+(-1)**(n-1)/(2*(n-1)+1)
    end do
    !$OMP end parallel do
    t2 = second()
    write(*,*) rank,result*4,t2-t1
    call mpi_finalize(ierr)
end program

program omp_mpi
    use omp_lib
    use mpi
    implicit none
    integer :: ierr, rank, procnum
    integer :: threadnum, id, ierr1, threads, i
    threadnum = 22
    call mpi_init(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD,rank,ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD,procnum,ierr)
    call omp_set_num_threads(threadnum)
    threads = omp_get_num_threads()
    !$OMP parallel do private(id, ierr1, threads) shared(threadnum,rank)
    do i = 1, threadnum
        id = omp_get_thread_num()
        ! threads = omp_get_num_threads()
        print*,id, 'of', threads, 'on CPU: ', rank
    end do
    !$OMP end parallel do
    call mpi_finalize(ierr)
end program


! program omp_mpi
!     use omp_lib
!     use mpi
!     implicit none
!     integer :: ierr, rank, procnum
!     integer :: threadnum, id, ierr1, threads, i
!     threadnum = 22
!     call mpi_init(ierr)
!     call MPI_COMM_RANK(MPI_COMM_WORLD,rank,ierr)
!     call MPI_COMM_SIZE(MPI_COMM_WORLD,procnum,ierr)
!     ! call omp_set_num_threads(threadnum)
!     ! threads = omp_get_num_threads()
!     !$OMP parallel num_threads(threadnum)
!     print*, rank
!     !$OMP end parallel
!     call mpi_finalize(ierr)
! end program